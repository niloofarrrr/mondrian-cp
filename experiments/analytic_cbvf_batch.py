"""Vectorized matched-policy evaluation; strict analytic QPs and exact flow.

This is an acceleration of analytic_cbvf_mechanism, not a different filter.
Truth parameters are read only for simulation and post-action diagnostics.
"""
import math
import numpy as np


def references(s, cfg, time):
    n=len(s);target=np.tile([cfg['waypoint_x'],.65],(n,1))
    routed=time>=cfg.get('route_switch_time',float('inf'))
    if routed:
        target[:,0]=cfg.get('route_waypoint_x',cfg['waypoint_x'])
        target[s[:,0]>cfg.get('route_gate_x',float('inf')),1]=cfg.get('route_waypoint_y',.65)
    target[s[:,1]>=.45]=[0.,1.8]
    direction=target-s[:,:2];r=np.linalg.norm(s[:,:2],axis=1)
    orbit=(s[:,1]<.45)&(r>0)&(r<cfg['orbit_trigger'])
    direction[orbit]=np.c_[s[orbit,1],-s[orbit,0]]/r[orbit,None]+2*(1-r[orbit,None])*s[orbit,:2]/r[orbit,None]
    error=(np.arctan2(direction[:,1],direction[:,0])-s[:,2]+math.pi)%(2*math.pi)-math.pi
    throttle=-1. if routed and time<cfg.get('route_switch_time',float('inf'))+cfg.get('reverse_duration',0.) else 1.
    return np.c_[np.full(n,throttle),np.clip(1.5*error/.8,-1,1)]


def flow(s,u,cfg,dt=None):
    dt=cfg['dt'] if dt is None else dt
    v=cfg['vmin']+(u[:,0]+1)*(.6-cfg['vmin'])/2;w=cfg['beta']*u[:,1]
    half=w*dt/2;distance=v*dt*np.sinc(half/math.pi)
    end=s.copy();end[:,0]+=distance*np.cos(s[:,2]+half);end[:,1]+=distance*np.sin(s[:,2]+half)
    end[:,2]=(s[:,2]+w*dt+math.pi)%(2*math.pi)-math.pi
    return end


def exact_clearance(s,u,cfg,end):
    v=cfg['vmin']+(u[:,0]+1)*(.6-cfg['vmin'])/2;w=cfg['beta']*u[:,1];dt=cfg['dt']
    minimum=np.minimum(np.linalg.norm(s[:,:2],axis=1),np.linalg.norm(end[:,:2],axis=1))-.7
    curved=np.abs(w)>=1e-10
    if np.any(curved):
        z=s[curved];vv=v[curved];ww=w[curved]
        cx=z[:,0]-vv/ww*np.sin(z[:,2]);cy=z[:,1]+vv/ww*np.cos(z[:,2])
        root=np.arctan2(-cx,cy)
        d=(root-z[:,2]+math.pi/2)%math.pi-math.pi/2
        t=d/ww;inside=(t>0)&(t<dt)
        if np.any(inside):
            indices=np.flatnonzero(curved)[inside]
            at=flow(s[indices],u[indices],cfg,t[inside])
            minimum[indices]=np.minimum(minimum[indices],np.linalg.norm(at[:,:2],axis=1)-.7)
    straight=~curved
    if np.any(straight):
        ids=np.flatnonzero(straight);z=s[ids];vel=v[ids,None]*np.c_[np.cos(z[:,2]),np.sin(z[:,2])]
        den=np.sum(vel*vel,axis=1);t=np.clip(-np.sum(z[:,:2]*vel,axis=1)/np.maximum(den,1e-300),0,dt)
        minimum[ids]=np.minimum(minimum[ids],np.linalg.norm(z[:,:2]+t[:,None]*vel,axis=1)-.7)
    return minimum


def evaluate(starts,cfg,method,buffers=(0.,0.),residual=None,trace_stride=1):
    if method not in ('nominal','cbvf','cp'):raise ValueError(method)
    if not (-.2<=cfg['vmin']<=.6 and 0<=cfg['beta']<=.8):raise ValueError('Dynamics outside declared score/envelope domain.')
    s=np.asarray(starts,dtype=float).copy();n=len(s);dt=cfg['dt'];gamma=cfg['gamma']
    if np.any(np.linalg.norm(s[:,:2],axis=1)<=.7):raise ValueError('Unsafe initial condition.')
    alive=np.ones(n,bool);unsafe=np.zeros(n,bool);goals=np.zeros(n,bool);steps=np.zeros(n,int)
    interventions=np.zeros(n,int);active_steps=np.zeros(n,int);violations=np.zeros(n,int)
    minimum=np.linalg.norm(s[:,:2],axis=1)-.7;minqp=np.full(n,np.inf);scores=np.zeros((n,2));visits=np.zeros((n,2),int)
    returns=np.zeros(n);traces=[];checks=0;minconstraint=np.inf;min_true_interval=np.inf
    delta=cfg['vmin']+.2;buffer=np.asarray(buffers)
    for j in range(round(cfg['duration']/dt)):
        ids=np.flatnonzero(alive)
        if not len(ids):break
        z=s[ids];r=np.linalg.norm(z[:,:2],axis=1);h=r-.7
        heading=np.c_[np.cos(z[:,2]),np.sin(z[:,2])];c=np.sum(z[:,:2]/r[:,None]*heading,axis=1)
        L=np.hypot(gamma+.6/(r-.6*dt),.6);eps=2*L*dt
        region=np.c_[h-dt<=.75,h+dt>=.75]
        xi=np.max(np.where(region,buffer[None,:],-np.inf),axis=1)
        eta=np.maximum(0,-delta*c)+abs(delta)*np.hypot(1/(r-.6*dt),1)*dt
        scores[ids]=np.maximum(scores[ids],np.where(region,eta[:,None],0));visits[ids]+=region
        nom=references(z,cfg,j*dt)
        if residual is not None:nom=np.clip(nom+cfg.get('residual_scale',.02)*residual(z),-1,1)
        u=nom.copy();base=gamma*h+.2*c;a=.4*c
        rhs=eps+(xi if method=='cp' else 0);available=base+np.abs(a)-rhs
        active=h<=cfg.get('activation_threshold',1.)
        if method!='nominal':
            if np.any(available[active]<-1e-10):
                bad=np.flatnonzero(active&(available<-1e-10))[0]
                raise RuntimeError('Strict QP infeasible before action: '+str(dict(method=method,episode=int(ids[bad]),step=j,state=z[bad].tolist(),margin=float(available[bad]))))
            project=active&(base+a*u[:,0]<rhs)
            if np.any(project&(np.abs(a)<1e-14)):raise RuntimeError('No projection authority.')
            u[project,0]=(rhs[project]-base[project])/a[project]
            if np.max(np.abs(u))>1+1e-8:raise RuntimeError('Actuator constraint violated.')
            minqp[ids]=np.minimum(minqp[ids],np.where(active,available,np.inf));active_steps[ids]+=active
        psi=base+a*u[:,0]
        v=cfg['vmin']+(u[:,0]+1)*(.6-cfg['vmin'])/2;truepsi=gamma*h+c*v
        interventions[ids]+=np.linalg.norm(u-nom,axis=1)>1e-8
        violations[ids]+=(psi>=eps-1e-9)&(truepsi<0)
        end=flow(z,u,cfg);localmin=exact_clearance(z,u,cfg,end);minimum[ids]=np.minimum(minimum[ids],localmin)
        hit=localmin<0;goal=np.linalg.norm(end[:,:2]-[0.,1.8],axis=1)<=.5
        # Time-integrated progress reward, plus terminal task outcomes; common
        # across methods and explicitly different from the old trainer reward.
        returns[ids]+=-dt*np.linalg.norm(end[:,:2]-[0.,1.8],axis=1)+100*goal-100*hit
        if method!='nominal' and np.any(active):
            minconstraint=min(minconstraint,float(np.min((psi-rhs)[active])));checks+=int(np.sum(active))
            min_true_interval=min(min_true_interval,float(np.min((truepsi-L*dt)[active])))
        if ids[0]==0 and (j%trace_stride==0 or hit[0] or goal[0]):
            traces.append([j,j*dt,*z[0],*end[0],*nom[0],*u[0],h[0],localmin[0],psi[0],truepsi[0],xi[0],L[0],eps[0],available[0],float(active[0]),truepsi[0]-L[0]*dt])
        s[ids]=end;steps[ids]+=1;unsafe[ids]|=hit;goals[ids]|=goal&~hit;alive[ids[hit|goal]]=False
    rr=[]
    for i in range(n):
        rr.append(dict(unsafe=bool(unsafe[i]),goal=bool(goals[i]),minimum=float(minimum[i]),steps=int(steps[i]),interventions=int(interventions[i]),active_steps=int(active_steps[i]),min_qp=None if method=='nominal' else float(minqp[i]),true_derivative_violations=int(violations[i]),scores=scores[i].tolist(),region_visits=visits[i].tolist(),episode_return=float(returns[i])))
    return dict(unsafe_rate=float(unsafe.mean()),goal_rate=float(goals.mean()),rollouts=rr,trace=traces,interval_checks=checks,min_constraint_margin=None if method=='nominal' else minconstraint,min_true_interval_margin=None if method=='nominal' else min_true_interval)


TRACE_HEADER='step,time,x,y,theta,next_x,next_y,next_theta,nom0,nom1,u0,u1,h,min_interval_h,psi_model,psi_true,xi,L,epsilon,feasibility,active,true_interval_lower_bound'

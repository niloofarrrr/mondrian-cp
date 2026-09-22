"""Design-only exact-CBVF mechanism experiment for the reversible unicycle.

On h >= 0, B=h is the exact discounted safety value: t=0 bounds the value
above by h, while the admissible zero-speed control holds position and
attains h. This removes table/derivative interpolation as a confound.
No learned actor or final evaluation seed is used by this initial screen.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def reference(s, cfg, time=0.):
    waypoint_x=cfg.get('route_waypoint_x',cfg['waypoint_x']) if time>=cfg.get('route_switch_time',float('inf')) else cfg['waypoint_x']
    target=np.array([waypoint_x,.65]) if s[1]<.45 else np.array([0.,1.8])
    if time>=cfg.get('route_switch_time',float('inf')) and s[1]<.45 and s[0]>cfg.get('route_gate_x',float('inf')):
        target=np.array([waypoint_x,cfg.get('route_waypoint_y',.65)])
    direction=target-s[:2];r=np.linalg.norm(s[:2])
    if s[1]<.45 and 0<r<cfg['orbit_trigger']:
        direction=np.array([s[1],-s[0]])/r+2*(1-r)*s[:2]/r
    error=(math.atan2(direction[1],direction[0])-s[2]+math.pi)%(2*math.pi)-math.pi
    throttle=-1. if cfg.get('route_switch_time',float('inf'))<=time<cfg.get('route_switch_time',float('inf'))+cfg.get('reverse_duration',0.) else 1.
    return np.array([throttle,np.clip(1.5*error/.8,-1.,1.)])


def fields(s,cfg):
    r=float(np.linalg.norm(s[:2]));n=s[:2]/r;heading=np.array([math.cos(s[2]),math.sin(s[2])])
    h=r-.7;c=float(n@heading)
    # Public common physical envelope v_abs <= .6, omega_abs <= .8;
    # neither the true beta nor a calibrated statistic enters this bound.
    rlow=r-.6*cfg['dt']
    if rlow<=0:raise RuntimeError('Reachable tube intersects nonsmooth obstacle center.')
    L=math.hypot(cfg['gamma']+.6/rlow,.6)
    epsilon=L*2*cfg['dt']
    return h,c,L,epsilon


def project(s,nom,cfg,buffer):
    h,c,L,epsilon=fields(s,cfg);base=cfg['gamma']*h+.2*c;a=.4*c
    rhs=epsilon+buffer;available=base+abs(a)-rhs
    if available<-1e-10:raise RuntimeError('Strict QP infeasible; no fallback executed: '+str((s.tolist(),available,rhs)))
    u=nom.copy()
    if base+a*u[0]<rhs:
        if abs(a)<1e-14:raise RuntimeError('No speed-control authority.')
        u[0]=(rhs-base)/a
    if max(abs(u))>1+1e-8:raise RuntimeError('Projection violates actuator bounds.')
    return u,available,L,epsilon,base+a*u[0]


def step_exact(s,u,cfg):
    v=cfg['vmin']+(u[0]+1)*(.6-cfg['vmin'])/2;w=cfg['beta']*u[1];dt=cfg['dt']
    def advance(t):
        z=s.copy()
        if abs(w)<1e-12:z[:2]+=t*v*np.array([math.cos(s[2]),math.sin(s[2])])
        else:
            z[0]+=v/w*(math.sin(s[2]+w*t)-math.sin(s[2]));z[1]+=v/w*(math.cos(s[2])-math.cos(s[2]+w*t))
        z[2]=(s[2]+w*t+math.pi)%(2*math.pi)-math.pi
        return z
    # Exact minimum clearance on a constant-control circular arc, including
    # all radial stationary angles. A straight segment uses its projection.
    times=[0.,dt]
    if abs(w)<1e-12:
        vel=v*np.array([math.cos(s[2]),math.sin(s[2])]);den=float(vel@vel)
        if den>0:times.append(float(np.clip(-s[:2]@vel/den,0,dt)))
    else:
        center=s[:2]+v/w*np.array([-math.sin(s[2]),math.cos(s[2])])
        phi=math.atan2(-center[0],center[1])
        for k in range(-4,5):
            t=(phi+k*math.pi-s[2])/w
            if 0<t<dt:times.append(t)
    minimum=min(float(np.linalg.norm(advance(t)[:2])-.7) for t in times)
    return advance(dt),minimum


def rollout(start,cfg,method,buffers=(0.,0.),score=False):
    s=np.array(start);minimum=float(np.linalg.norm(s[:2])-.7);scores=np.zeros(2);min_qp=float('inf');ints=0
    true_violation=0;rows=[]
    for j in range(round(cfg['duration']/cfg['dt'])):
        nom=reference(s,cfg,j*cfg['dt']);h,c,L,epsilon=fields(s,cfg)
        regions=[i for i,(lo,hi) in enumerate([(-float('inf'),.75),(.75,float('inf'))]) if h+cfg['dt']>=lo and h-cfg['dt']<=hi]
        xi=max(buffers[i] for i in regions)
        if score:
            dv=cfg['vmin']+.2
            eta=max(0.,-dv*c)
            # Uniform control mismatch; an analytic trajectory-Lipschitz
            # margin covers every continuous interval, not just samples.
            pad=abs(dv)*math.hypot(1/(h+.7-.6*cfg['dt']),1)*cfg['dt']
            for i in regions:scores[i]=max(scores[i],eta+pad)
        if method=='nominal':u=nom;available=float('inf');psi=cfg['gamma']*h+c*(.2+.4*u[0])
        else:
            u,available,L,epsilon,psi=project(s,nom,cfg,xi if method=='cp' else 0.)
            min_qp=min(min_qp,available)
        ints+=np.linalg.norm(u-nom)>1e-8
        vtrue=cfg['vmin']+(u[0]+1)*(.6-cfg['vmin'])/2
        truepsi=cfg['gamma']*h+c*vtrue
        true_violation+=psi>=epsilon-1e-9 and truepsi<0
        before=s.copy();s,localmin=step_exact(s,u,cfg);minimum=min(minimum,localmin)
        if j%20==0:rows.append([j,*before,*nom,*u,h,psi,truepsi,xi,L,epsilon,available])
        unsafe=minimum<0;goal=np.linalg.norm(s[:2]-[0,1.8])<=.5
        if unsafe or goal:break
    return dict(unsafe=bool(unsafe),goal=bool(goal),minimum=minimum,steps=j+1,interventions=int(ints),
                min_qp=None if method=='nominal' else min_qp,true_derivative_violations=int(true_violation),
                scores=scores.tolist(),trace=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--beta',type=float,default=.43);p.add_argument('--vmin',type=float,default=-.15)
    p.add_argument('--orbit',type=float,default=.825);p.add_argument('--waypoint-x',type=float,default=-.9);p.add_argument('--gamma',type=float,default=1.5)
    p.add_argument('--duration',type=float,default=12.);p.add_argument('--route-switch-time',type=float,default=float('inf'));p.add_argument('--route-waypoint-x',type=float,default=-1.4)
    p.add_argument('--route-waypoint-y',type=float,default=.65);p.add_argument('--route-gate-x',type=float,default=float('inf'))
    p.add_argument('--initial-theta',type=float,default=math.pi/4);p.add_argument('--reset-jitter',type=float,default=.1)
    p.add_argument('--reverse-duration',type=float,default=0.)
    p.add_argument('--seed',type=int,default=810001);p.add_argument('--episodes',type=int,default=10);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cfg=dict(beta=a.beta,vmin=a.vmin,orbit_trigger=a.orbit,waypoint_x=a.waypoint_x,gamma=a.gamma,dt=.002,duration=a.duration,route_switch_time=a.route_switch_time,route_waypoint_x=a.route_waypoint_x)
    cfg.update(route_waypoint_y=a.route_waypoint_y,route_gate_x=a.route_gate_x,initial_theta=a.initial_theta,reset_jitter=a.reset_jitter)
    cfg['reverse_duration']=a.reverse_duration
    (out/'configuration.json').write_text(json.dumps(dict(configuration=cfg,seed=a.seed,design_only=True,actor='fixed shared reference; no training'),indent=2))
    rng=np.random.default_rng(a.seed);center=np.array([-.9,-3,a.initial_theta]);calstarts=center+rng.uniform(-a.reset_jitter,a.reset_jitter,(78,3))
    cal=[]
    for i,s in enumerate(calstarts):
        r=rollout(s,cfg,'cbvf',score=True);r.pop('trace');cal.append(r)
        if i%10==0:print('calibration',i,flush=True)
    scores=np.array([r['scores'] for r in cal]);buffers=scores.max(axis=0)
    (out/'calibration.json').write_text(json.dumps(dict(buffers=buffers.tolist(),rollouts=cal),indent=2))
    starts=center+np.random.default_rng(a.seed+10000).uniform(-a.reset_jitter,a.reset_jitter,(a.episodes,3))
    result={}
    for method in ['nominal','cbvf','cp']:
        rr=[]
        for i,s in enumerate(starts):
            r=rollout(s,cfg,method,buffers);trace=r.pop('trace')
            if i==0:np.savetxt(out/(method+'_trace.csv'),trace,delimiter=',',header='step,x,y,theta,nom0,nom1,u0,u1,h,psi_model,psi_true,xi,L,epsilon,feasibility',comments='')
            rr.append(r)
        result[method]=dict(unsafe_rate=float(np.mean([r['unsafe'] for r in rr])),goal_rate=float(np.mean([r['goal'] for r in rr])),rollouts=rr)
        (out/'results.json').write_text(json.dumps(result,indent=2));print(method,result[method]['unsafe_rate'],result[method]['goal_rate'],flush=True)
    print('SCREEN ONLY: fresh coverage, independent replication and exhaustive audits still required.',flush=True)


if __name__=='__main__':main()

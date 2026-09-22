"""Train one nominal-model residual actor, shared by every test condition/filter."""
import argparse
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gymnasium as gym
import numpy as np
from jaxrl5.agents import SACLagLearner
from jaxrl5.agents.agent import save_agent
from jaxrl5.data import ReplayBuffer
from analytic_cbvf_mechanism import reference,step_exact


DEFAULTS=dict(algorithm='sac_lag',max_steps=20000,start_training=1000,reference_proposal_steps=1000,batch_size=256,hidden_dims=[256,256],actor_lr=.0003,critic_lr=.0003,temp_lr=.0003,lag_lr=.0003,discount=.99,tau=.005,num_qs=2,num_min_qs=None,critic_dropout_rate=None,critic_layer_norm=False,target_entropy=-1.,init_temperature=.1,init_lag=0.,backup_entropy=True,cost_limit=1.,utd_ratio=1,residual_scale=.02,replay_capacity=20000,training_condition='aligned',training_filter='nominal',observation='[x,y,cos(theta),sin(theta),goal_x,goal_y]; time-dependent shared reference is composed outside residual network',reward='-dt*distance_to_goal(next_state)+100*goal-100*collision',cost='1 on exact-arc collision, 0 otherwise',termination='exact-arc collision or endpoint goal; horizon truncation bootstraps',logging_interval=1000)


def obs(s):return np.array([s[0],s[1],np.cos(s[2]),np.sin(s[2]),0.,1.8],dtype=np.float32)


def train(out,cfg,seed,training=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);args=dict(DEFAULTS);args.update(training or {})
    cfg=dict(cfg,beta=.8,vmin=-.2)
    (out/'training_configuration.json').write_text(json.dumps(dict(seed=seed,configuration=cfg,training=args),indent=2)+'\n')
    obs_space=gym.spaces.Box(-np.inf,np.inf,(6,),dtype=np.float32);action_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
    obs_space.seed(seed);action_space.seed(seed+1)
    keys=['actor_lr','critic_lr','temp_lr','lag_lr','discount','tau','num_qs','num_min_qs','critic_dropout_rate','critic_layer_norm','target_entropy','init_temperature','init_lag','backup_entropy','cost_limit']
    agent=SACLagLearner.create(seed,obs_space,action_space,hidden_dims=tuple(args['hidden_dims']),**{k:args[k] for k in keys})
    replay=ReplayBuffer(obs_space,action_space,args['replay_capacity']);replay.seed(seed+2)
    rng=np.random.default_rng(seed+100000);center=np.array([-.9,-3,cfg['initial_theta']]);jitter=cfg['reset_jitter']
    s=center+rng.uniform(-jitter,jitter,3);ep_steps=0;episode=0;ep_return=0.;ep_min=float('inf');unsafe_count=0;goals=0;start=time.monotonic()
    with (out/'training_episodes.jsonl').open('w') as epfile,(out/'training_progress.jsonl').open('w') as progress:
        for step in range(1,args['max_steps']+1):
            observation=obs(s)
            if step<=args['reference_proposal_steps']:residual=np.zeros(2,dtype=np.float32)
            else:residual,agent=agent.sample_actions(observation)
            nominal=np.clip(reference(s,cfg,ep_steps*cfg['dt'])+args['residual_scale']*np.asarray(residual),-1.,1.)
            next_s,minimum=step_exact(s,nominal,cfg);unsafe=minimum<0;goal=np.linalg.norm(next_s[:2]-[0,1.8])<=.5
            ep_steps+=1;terminated=bool(unsafe or goal);done=terminated or ep_steps>=round(cfg['duration']/cfg['dt'])
            reward=-cfg['dt']*np.linalg.norm(next_s[:2]-[0,1.8])+100*goal-100*unsafe
            replay.insert(dict(observations=observation,actions=np.asarray(residual,dtype=np.float32),rewards=float(reward),costs=float(unsafe),masks=float(not terminated),dones=done,next_observations=obs(next_s)))
            ep_return+=reward;ep_min=min(ep_min,minimum);s=next_s
            if step>=args['start_training']:
                batch=replay.sample(args['batch_size']*args['utd_ratio']);agent,info=agent.update(batch,args['utd_ratio'])
            if done:
                episode+=1;unsafe_count+=int(unsafe);goals+=int(goal and not unsafe)
                epfile.write(json.dumps(dict(step=step,episode=episode,episode_return=ep_return,steps=ep_steps,unsafe=bool(unsafe),goal=bool(goal and not unsafe),minimum=ep_min))+'\n');epfile.flush()
                s=center+rng.uniform(-jitter,jitter,3);ep_steps=0;ep_return=0.;ep_min=float('inf')
            if step%args['logging_interval']==0:
                row=dict(step=step,episodes=episode,unsafe_episodes=unsafe_count,goal_episodes=goals,elapsed_seconds=time.monotonic()-start)
                progress.write(json.dumps(row)+'\n');progress.flush();print(json.dumps(row),flush=True)
    save_agent(agent,str(out),args['max_steps'])
    result=dict(completed=True,seed=seed,steps=args['max_steps'],episodes=episode,unsafe_episodes=unsafe_count,goal_episodes=goals,elapsed_seconds=time.monotonic()-start)
    (out/'training_complete.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',type=Path,required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    data=json.loads(a.configuration.read_text());train(a.out,data['configuration'],a.seed,data.get('training'))

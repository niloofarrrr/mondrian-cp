"""Design-only CPU timing; never used as training or qualification evidence."""
import sys,time,json,hashlib,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import gymnasium as gym
import jax
from jaxrl5.agents.sac_lag.sac_lag_learner import SACLagLearner
agent=SACLagLearner.create(610954,gym.spaces.Box(-np.inf,np.inf,(6,),dtype=np.float32),gym.spaces.Box(-1,1,(2,),dtype=np.float32),hidden_dims=(256,256))
rng=np.random.default_rng(610954)
batch={k:rng.normal(size=shape).astype(np.float32) for k,shape in [('observations',(256,6)),('next_observations',(256,6)),('actions',(256,2)),('rewards',(256,)),('costs',(256,))]};batch.update(masks=np.ones(256,np.float32),dones=np.zeros(256,np.float32))
def sync(a):
 for x in jax.tree_util.tree_leaves(a.actor.params):x.block_until_ready()
agent,_=agent.update(batch,utd_ratio=1);sync(agent)
start=time.perf_counter()
for _ in range(30):agent,_=agent.update(batch,utd_ratio=1)
sync(agent);elapsed=time.perf_counter()-start
checksum=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(agent.actor.params))).hexdigest()
print(json.dumps({'updates':30,'seconds':elapsed,'actor_sha256':checksum,'XLA_FLAGS':os.environ.get('XLA_FLAGS','default')}),flush=True)

import math
import unittest
import numpy as np
from analytic_cbvf_mechanism import fields,project,step_exact
from analytic_cbvf_mechanism import reference,rollout
from analytic_cbvf_batch import references,flow,exact_clearance,evaluate


class AnalyticCertificateTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(beta=.43,vmin=-.15,gamma=1.5,dt=.002)

    def test_projection_has_no_true_dynamics_or_calibration_leak(self):
        s=np.array([-.74,0.,0.]);u=np.array([1.,.4])
        a=project(s,u,self.cfg,0.)[0]
        b=project(s,u,dict(self.cfg,beta=.8,vmin=-.2),0.)[0]
        np.testing.assert_array_equal(a,b)
        tightened=project(s,u,self.cfg,.04)
        self.assertGreaterEqual(tightened[-1],tightened[3]+.04-1e-12)
        self.assertLess(tightened[0][0],a[0])

    def test_exact_flow_composes_and_arc_minimum_encloses_samples(self):
        rng=np.random.default_rng(910001)
        for _ in range(100):
            s=np.r_[rng.uniform(-2,2,2),rng.uniform(-math.pi,math.pi)]
            u=rng.uniform(-1,1,2)
            end,minimum=step_exact(s,u,self.cfg)
            middle,_=step_exact(s,u,dict(self.cfg,dt=.001))
            twice,_=step_exact(middle,u,dict(self.cfg,dt=.001))
            np.testing.assert_allclose(end,twice,atol=2e-12,rtol=0)
            sampled=[np.linalg.norm(step_exact(s,u,dict(self.cfg,dt=t))[0][:2])-.7 for t in np.linspace(0,.002,21)]
            self.assertLessEqual(minimum,min(sampled)+1e-12)

    def test_lipschitz_bound_encloses_analytic_gradients(self):
        rng=np.random.default_rng(910002)
        for _ in range(300):
            a=rng.uniform(-math.pi,math.pi);r=rng.uniform(.701,3.)
            s=np.array([r*math.cos(a),r*math.sin(a),rng.uniform(-math.pi,math.pi)])
            h,c,L,eps=fields(s,self.cfg);n=s[:2]/r;heading=np.array([math.cos(s[2]),math.sin(s[2])])
            v=rng.uniform(-.2,.6)
            spatial=v*(np.eye(2)-np.outer(n,n))@heading/r+1.5*n
            theta=v*n@np.array([-math.sin(s[2]),math.cos(s[2])])
            self.assertLessEqual(np.linalg.norm(np.r_[spatial,theta]),L+1e-12)
            self.assertAlmostEqual(eps,L*2*.002,places=14)

    def test_batch_flow_and_reference_match_independent_scalar(self):
        cfg=dict(self.cfg,orbit_trigger=.825,waypoint_x=-.9,route_switch_time=8.,route_waypoint_x=-1.4,route_waypoint_y=-1.2,route_gate_x=-1.2,reverse_duration=2.)
        rng=np.random.default_rng(910003);s=np.c_[rng.uniform(-2,2,(100,2)),rng.uniform(-math.pi,math.pi,100)];u=rng.uniform(-1,1,(100,2))
        end=flow(s,u,cfg);mins=exact_clearance(s,u,cfg,end)
        for i in range(100):
            scalar,minimum=step_exact(s[i],u[i],cfg)
            np.testing.assert_allclose(end[i],scalar,atol=2e-12,rtol=0)
            self.assertAlmostEqual(mins[i],minimum,places=11)
        for time in [0.,8.,9.9,10.]:
            np.testing.assert_allclose(references(s,cfg,time),np.array([reference(z,cfg,time) for z in s]),atol=1e-12,rtol=0)

    def test_batch_full_rollouts_match_scalar(self):
        cfg=dict(self.cfg,orbit_trigger=0.,waypoint_x=-.9,duration=25.,route_switch_time=8.,route_waypoint_x=-1.4,route_waypoint_y=-1.2,route_gate_x=-1.2,reverse_duration=2.)
        starts=np.array([[-.9,-3,.7],[-.91,-3.01,.69]])
        for method in ['nominal','cbvf','cp']:
            batch=evaluate(starts,cfg,method,[.05025,.05025],trace_stride=20)
            for s,b in zip(starts,batch['rollouts']):
                r=rollout(s,cfg,method,[.05025,.05025],score=True)
                for key in ['unsafe','goal','steps','interventions','true_derivative_violations']:
                    self.assertEqual(b[key],r[key],(method,key))
                self.assertAlmostEqual(b['minimum'],r['minimum'],places=10)
                np.testing.assert_allclose(b['scores'],r['scores'],atol=1e-11,rtol=0)

    def test_batch_uncal_action_ignores_truth_and_buffers(self):
        cfg=dict(self.cfg,orbit_trigger=0.,waypoint_x=-.9,duration=.002)
        start=np.array([[-.75,0.,0.]])
        baseline=evaluate(start,cfg,'cbvf',[0.,0.])['trace'][0][10:12]
        changed=evaluate(start,dict(cfg,beta=.8,vmin=-.2),'cbvf',[.1,.2])['trace'][0][10:12]
        np.testing.assert_array_equal(baseline,changed)
        cp=evaluate(start,cfg,'cp',[.04,.04])['trace'][0][10:12]
        self.assertLess(cp[0],baseline[0])


if __name__=='__main__':unittest.main()

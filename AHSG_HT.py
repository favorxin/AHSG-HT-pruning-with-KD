import torch
import numpy as np
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required
import random
#from ht import L1_norm

def L1_norm_resnet(group,rate):
    index_prun = {}
    #for idx, p in enumerate(group.parameters()):
    for idx, p in enumerate(group['params']):
        if idx%3==0 and len(p.size())==4:
            b = []
            prun = int(p.size()[0] * (rate))
            for k in p:
                #for k in p.abs().max(1)[0]:
                b.append(torch.norm(k, 2))
            b = torch.FloatTensor(b)
            b = b.cpu().numpy()
            index = b.argsort()[::-1][prun:]
            index_prun[idx]=index
    #print(index_prun)
    #for i, p in enumerate(group.parameters()):
    for i, p in enumerate(group['params']):
       if i == 0 and len(p.size())==4:
           p.data[index_prun[i].tolist(), :, :, :] = 0
       elif i % 3 == 0 and len(p.size())==4:
           #j[1].data[:, index_prun[i - 3].tolist(), :, :] = 0
           p.data[index_prun[i].tolist(), :, :, :] = 0
       if i%3==2 and len(p.size())==1:
           p.data[index_prun[i-2].tolist()]=0
       #print('i: {}, p.size: {}'.format(i, p.size()))
       #print('i: {}, p: {}'.format(i, p))

def L1_norm_vgg_bn(group, rate):
    index_prun_in = {}
    net_index = [52]  # [32,40,52,64]
    #net_nop_index = [28]   #[28,36,48,60]
    for idx, p in enumerate(group['params']):
        if idx == 0 and len(p.size())==4:
            prun_in = int(p.size()[0] * (rate))
            p_copy = p.data.clone().cpu().numpy()
            p_sum_in = np.sum(abs(p_copy), axis=(1,2,3))
            p_sort_in = np.argsort(p_sum_in)
            index_in = p_sort_in[::-1][prun_in:]            
            index_prun_in[idx] = index_in
            p.data[index_in.tolist(),:,:,:] = 0        
        elif idx % 4 ==0 and len(p.size())==4 and idx < net_index[0]:
            prun_in = int(p.size()[0] * (rate))
            prun_out = int(p.size()[1] * (rate))
            p_copy = p.data.clone().cpu().numpy()
            p_sum_in = np.sum(abs(p_copy), axis=(1,2,3))
            p_sort_in = np.argsort(p_sum_in)
            index_in = p_sort_in[::-1][prun_in:]            
            #mask = np.zeros(p.size()[0])
            #mask[index_in.tolist()] = 1
            index_prun_in[idx] = index_in
            #idx_in = np.squeeze(np.argwhere(np.asarray(mask)))
            p.data[index_in.tolist(),:,:,:] = 0
            
            p_copy = p.data.clone().cpu().numpy()
            p_sum_out = np.sum(abs(p_copy), axis=(0,2,3))
            p_sort_out = np.argsort(p_sum_out)
            index_out = p_sort_out[::-1][prun_out:]            
            #mask = np.zeros(p.size()[1])
            #mask[index_out.tolist()] = 1     
            #idx_out = np.squeeze(np.argwhere(np.asarray(mask)))
            p.data[:,index_out.tolist(),:,:] = 0
        elif idx == net_index[0]:
            prun_out = int(p.size()[1] * (rate))
            p_copy = p.data.clone().cpu().numpy()
            p_sum_out = np.sum(abs(p_copy), axis=(0))
            p_sort_out = np.argsort(p_sum_out)
            index_out = p_sort_out[::-1][prun_out:] 
            p.data[:,index_out.tolist()] = 0

    #print("index_prun_in:",index_prun_in) 
    for i, p in enumerate(group['params']):
        if i < net_index[0]:
#           if i % 3 == 0 and len(p.size())==4:
#               p.data[index_prun_in[i].tolist(),:,:,:] = 0
#               p.data[:,index_prun_out[i].tolist(),:,:] = 0
            if i%4==1 and len(p.size())==1:
                p.data[index_prun_in[i-1].tolist()]=0
            if i%4==2 and len(p.size())==1:
                p.data[index_prun_in[i-2].tolist()]=0
            if i%4==3 and len(p.size())==1:
                p.data[index_prun_in[i-3].tolist()]=0

class AHSG_HT(Optimizer):
    r"""Implements stochastic variance reduced gradient or HSG_HT (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        snapshot_params (iterable): iterable of parameters for the snapshot model or dicts
            defining parameter groups in the same way as params
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        update_frequency (int, optional): determines after how many epochs the snapshot should be updated (default: 1)
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, v=0, HTrate=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, v=v, HTrate=HTrate)
        super(AHSG_HT, self).__init__(params, defaults)

#        # Store the update_frequency parameter       , update_frequency=1    , snapshot_params=None


#        # Add the full gradient to the parameter groups
#        for idx, group in enumerate(self.param_groups):
#            group['full_gradient'] = list()
#            for p in group['params']:
#                group['full_gradient'].append(torch.zeros_like(p.data))

        # Add the previous data to the parameter groups
        for idx, group in enumerate(self.param_groups):
            group['prev_data'] = list()
            for p in group['params']:
                group['prev_data'].append(torch.zeros_like(p.data.cuda()))

    def step(self, closure=None, k=None, epoch=2, final=0):
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss, snapshot_loss = None, None
        if closure is not None:
            loss, snapshot_loss = closure()
            
        if k is not None:
            s_sum = k 
        
        if final != 0:
            final=1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            HTrate = group['HTrate']
            v = group['v']
            
            for idx, p in enumerate(group['params']):
                #print('idx: {:03d}; p.size: {}'.format(idx, p.size()))
            #for p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("HSG_HT doesn't support sparse gradients")
                    
                
                #average_gradient = group['full_gradient'][idx]/s_sum
                #average_gradient = group['full_gradient'][idx]
                prev_data = group['prev_data'][idx]
                p_v = p.data - prev_data
                # gradient data             
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
                p.data.add_(v,p_v)
                
#                print('p.data.size:',p.data.size())
#                print("nei.1")
            #print(epoch)
            #if epoch%3==0:            
#            if final == 1:
#                start = time.time()
#                #L1_norm_vgg(group,HTrate)
#                #L1_norm_vgg_bn(group,HTrate)
#                L1_norm_resnet(group,HTrate)
#                end = time.time() - start
#                #L1_norm_resnet(group,HTrate)            
#                print('Pruning complete in {:.0f}m {:.0f}s:'.format(end // 60, end % 60))
#                print("time:",end)
            
            L1_norm_resnet(group,HTrate)
            #L1_norm_vgg_bn(group,HTrate)
                
        return loss

    def update_HSG(self, zero1):  
        r"""Updates the parameter snapshot and the average gradient
        Arguments:
            dataloader : A dataloader used to get the training samples.
            closure (callable): A closure that reevaluates the snapshot model
                and returns the loss.
        """

        if zero1 is None:
            raise RuntimeError("Zero1 has to be given")

        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if zero1==1:
                    #group['full_gradient'][idx].zero_() 
                    group['prev_data'][idx].zero_()                   
                #group['full_gradient'][idx].add_(1, p.grad.data)
                group['prev_data'][idx].add_(1, p.data)

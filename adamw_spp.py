import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import numpy as np 


class AdamWSPP(Optimizer):
    r"""Implements AdamW algorithm.

    Need to reformate for lbi
    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        mu (float, optional): the mu parameter in loss function between W and Gamma
        kappa (float, optional): kappa parameter in W update rule.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, mu=1, kappa=1, lamda=0.1,betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-1, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False,
                 namelist=None, gamma_result_save_path: str='output/deit_s/prox_wd_gamma_deit_s.pth'):
        print(f"Adamw prox setting: \n gamma_result_save_path: {gamma_result_save_path}")
        self.gamma_result_save_path = gamma_result_save_path
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, mu=mu, kappa=kappa, lamda=lamda, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(AdamWSPP, self).__init__(params, defaults)
        # print(self.param_groups['params'])

    def assign_name(self, name_list):
        pass

    def initialize_prox(self, layer_list):
        pass

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def check_sparsity(self, recorder=None):
        N = 0
        N_s = 0
        torch.cuda.empty_cache()
        for group in self.param_groups:
           for p in group['params']:
                param_state = self.state[p]
                N += p.size().numel()
                N_s += torch.gt(torch.abs(p.data), 0.0).float().sum()
        print('Sparsity :' , float(N_s) / float(N))
        if recorder:
            recorder.write('Checking Sparsity :')
            recorder.write(str(float(N_s) / float(N)))
            recorder.write('\n')
            recorder.flush()
        return float(N_s) / float(N)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        sum_all_attn_row = 0.0
        sum_pos_attn_row = 0.0
        sum_all_mlp_row = 0.0
        sum_pos_mlp_row = 0.0
        sparsity_attn=0
        sparsity_mlp=0

        # for group in self.param_groups:
        #     for p in group['params']:
        #         param_state = self.state[p]
        #         if len(param_state) > 1:
        #             if ('alpha' in param_state['name']) :

        #                 if 'attn' in param_state['name'] :                                
        #                     # sum_all_attn_row += param_state['z_buffer'].numel()
        #                     # sum_pos_attn_row += torch.count_nonzero(torch.abs(param_state['z_buffer'])>1).item()
        #                     sum_all_attn_row += param_state['gamma_buffer'].numel()
        #                     sum_pos_attn_row += torch.count_nonzero(torch.abs(param_state['gamma_buffer'])).item()


        #                 if 'mlp' in param_state['name'] :                        
        #                     # sum_all_mlp_row += param_state['z_buffer'].numel()
        #                     # sum_pos_mlp_row += torch.count_nonzero(torch.abs(param_state['z_buffer'])>1).item()
        #                     sum_all_mlp_row += param_state['gamma_buffer'].numel()
        #                     sum_pos_mlp_row += torch.count_nonzero(torch.abs(param_state['gamma_buffer'])).item()

        # if len(param_state) > 1:
        #     sparsity_attn = sum_pos_attn_row / sum_all_attn_row
        #     # sparsity_attn_col = sum_pos_attn_col / sum_all_attn_col
        #     sparsity_mlp = sum_pos_mlp_row / sum_all_mlp_row

        #sparsity_attn,sparsity_mlp,_=self.print_sparsity_all(print=None)
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            gamma_buffers = []
            z_buffers = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            mu, kappa,lamda= group['mu'], group['kappa'],group['lamda']
            name_list=[]

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if  ('alpha' in state['name']):
                        #print("set_gamma")
                        state['gamma_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        state['z_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        state['step_record'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['gamma_buffer'] = torch.zeros([1])

                        state['z_buffer'] = torch.zeros([1])
                        state['step_record'] = torch.zeros([1])
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                name_list.append(state['name'])
                gamma_buffers.append(state['gamma_buffer'])
                z_buffers.append(state['z_buffer'])
                
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state_steps.append(state['step'])

            adamwprox(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  gamma_buffers,
                  z_buffers,
                  sparsity_attn,
                  sparsity_mlp,
                  state_steps,
                  amsgrad=amsgrad,
                  mu=mu,
                  lamda=lamda,
                  kappa=kappa,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'],
                  name_list=name_list
                  )

        return loss

    def print_sparsity_all(self,print=True):
        with torch.no_grad():
            sum_all_attn_row = 0.0
            sum_pos_attn_row = 0.0
            z_max_attn_row = 0
            sum_all_mlp_row = 0.0
            sum_pos_mlp_row = 0.0
            z_max_attn_col=0.0
            z_max_mlp_col=0.0
            

            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    #print("group ipdate\n\n")
                    #print(param_state['name'])
                    if ('alpha' in param_state['name']) :
                        if ('_m' not in param_state['name']):
                            #print("alpha ipdate\n")
                            if 'attn' in param_state['name'] :

                                #print("attn ipdate\n")
                                #print(param_state.keys())
                                sum_all_attn_row += param_state['gamma_buffer'].numel()
                                #if param_state['gamma_buffer'] !=0:
                                sum_pos_attn_row += torch.count_nonzero(torch.abs(param_state['gamma_buffer']))
                                param_state['step_record'][param_state['gamma_buffer']!=0] +=1
                                #    sum_pos_attn_row +=1
                            # if 'attn.proj.weight' in param_state['name'] :
                                if param_state['z_buffer'].max().item() > z_max_attn_col:
                                    z_max_attn_col = param_state['z_buffer'].max().item()
                                
                            #     sum_all_attn_col += param_state['gamma_buffer'].numel()
                            #     sum_pos_attn_col += torch.count_nonzero(param_state['gamma_buffer']).item()
                            if 'mlp' in param_state['name'] :
                                
                                sum_all_mlp_row += param_state['gamma_buffer'].numel()
                                sum_pos_mlp_row +=  torch.count_nonzero(torch.abs(param_state['gamma_buffer']))
                                param_state['step_record'][param_state['gamma_buffer']!=0] +=1
                                # if param_state['gamma_buffer'] !=0:
                                #     sum_pos_mlp_row += 1
                            # if '.mlp.fc2' in param_state['name'] :
                                if param_state['z_buffer'].max().item() > z_max_mlp_col:
                                    z_max_mlp_col = param_state['z_buffer'].max().item()
                                
                            #     sum_all_mlp_col += param_state['gamma_buffer'].numel()
                            #     sum_pos_mlp_col += torch.count_nonzero(param_state['gamma_buffer']).item()
                            # if 'alpha' in param_state['name'] :
                            #     if param_state['z_buffer'].max().item() > z_max_alpha:
                            #         z_max_alpha = param_state['z_buffer'].max().item()
                                
                            #     sum_all_alpha += param_state['gamma_buffer'].numel()
                            #     sum_pos_alpha += torch.count_nonzero(param_state['gamma_buffer']).item()

            sparsity_attn_row = sum_pos_attn_row / sum_all_attn_row
            # sparsity_attn_col = sum_pos_attn_col / sum_all_attn_col
            sparsity_mlp_row = sum_pos_mlp_row / sum_all_mlp_row
            # sparsity_mlp_col = sum_pos_mlp_col / sum_all_mlp_col
            # sparsity_alpha = sum_pos_alpha / sum_all_alpha
            sparsity_all = (sum_pos_attn_row+sum_pos_mlp_row) / (sum_all_attn_row+sum_all_mlp_row)
            # sparsity_all = (sum_pos_attn_row+sum_pos_attn_col+sum_pos_mlp_row+sum_pos_mlp_col+sum_pos_alpha) / (sum_all_attn_row+sum_all_attn_col+sum_all_mlp_row+sum_all_mlp_col+sum_all_alpha)
            #print("all counts: ", sum_all)
            #print("all sum_pos: ", sum_pos)
            # print("z_max_mlp_row: ", z_max_mlp_row)
            # print("z_max_mlp_col_out: ", z_max_mlp_col)
            # # print("z_max_attn_row: ", z_max_attn_row)
            # print("z_max_attn_col_out: ", z_max_attn_col)
            # print("z_max_alpha: ", z_max_alpha)
            # if print:
            #     # print("current sparsity_attn_row: ", sparsity_attn_row)

            #     # print("current sparsity_mlp_row: ", sparsity_mlp_row)
            #     # # print("current sparsity_mlp_col_out: ", sparsity_mlp_col)
            #     # # print("current sparsity_attn_col_out: ", sparsity_attn_col)
            #     # # print("current sparsity_alpha: ", sparsity_alpha)
            #     # print("current sparsity_all: ", sparsity_all)
            #     print("current sparsity_attn_row: ", sum_all_attn_row )

            #     print("current sparsity_mlp_row: ", sum_pos_mlp_row)
                # print("current sparsity_mlp_col_out: ", sparsity_mlp_col)
                # print("current sparsity_attn_col_out: ", sparsity_attn_col)
                # print("current sparsity_alpha: ", sparsity_alpha)
                #print("current sparsity_all: ", sparsity_all)



            return sparsity_attn_row,sparsity_mlp_row,sparsity_all,z_max_attn_col,z_max_mlp_col



def shrink(s_t, lam):
    #proximal mapping for 2-d weight(fc layer)
    gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
    return gamma_t


def shrink_group(ts):
    # shrinkage for 4-d weight(conv layer)
    ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
    ts_norm = torch.norm(ts_reshape,2,1)
    ts_shrink = torch.max(torch.zeros_like(ts_norm),torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm),ts_norm))
    ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape,0,1),ts_shrink),0,1)
    ts_return = torch.reshape(ts_return,ts.shape)
    return ts_return


def adamwprox(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          gamma_buffers: List[Tensor],
          z_buffers: List[Tensor],
          sparsity_attn: float,
          sparsity_mlp: float,
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          mu: float,
          kappa: float,
          lamda: float,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool,
          name_list:List[str]
          ):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamwprox
    else:
        func = _single_tensor_adamwprox

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         gamma_buffers,
         z_buffers,
         sparsity_attn,
         sparsity_mlp,
         state_steps,
         amsgrad=amsgrad,
         mu=mu,
         kappa=kappa,
         lamda=lamda,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         name_list=name_list
         )


def _single_tensor_adamwprox(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         gamma_buffers: List[Tensor],
                         z_buffers: List[Tensor],
                         sparsity_attn: float,
                         sparsity_mlp: float,
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         mu: float,
                         kappa: float,
                         lamda: float,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool,
                         name_list: List[str]
                         ):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        gamma_buffer = gamma_buffers[i]
        z_buffer = z_buffers[i]
        lbi_cases = [2] 
        name=name_list[i]

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        # update step
        step_t += 1
        #print(step_t)

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        #print("capturable",capturable)
        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            #print("cap\n")
            step = step_t.item()
            #step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            
            param.addcdiv_(exp_avg, denom, value=-step_size)

            if 'alpha' in name:
                #gamma_buffer.addcdiv_(exp_avg, 1, value=-step_size)
                total_steps=1000*20
                #print(sparsity_attn,sparsity_mlp)
                if 'attn' in name:
                    kappa=1
                    #if sparsity_attn>0.68:
                    z_buffer.add_(step_size*kappa*(param-gamma_buffer))
                if 'mlp' in name:
                    kappa=1
                    #if sparsity_mlp>0.68:
                    z_buffer.add_(step_size*kappa*(param-gamma_buffer))
        #total_steps=625*10
        if 'alpha' in name:
            #lamda=(step_t+1)/total_steps
            # if step_t>total_steps:
            #     lamda=(step_t+1)/total_steps
            # else:
            #     #lamda=((1-math.cos(math.pi*(step_t+1)/total_steps))/2)**(1/2)
            #     lamda=(step_t+1)/total_steps
            lamda=3#small 15
            g_sign = (1 - lamda/ torch.abs(z_buffer)).relu()
            #g_sign = -((-g_sign * z_buffer).relu())
            g_sign = ((g_sign * z_buffer)).relu()
            if sparsity_mlp<1.68:
                gamma_buffer.copy_(g_sign)



        #     s=1
        #     lamda=1
        #     total_steps=417*20
        #     lamda_k=1
        #     if step_t<total_steps:
        #         lamda_k=((1-math.cos(math.pi*(step_t+1)/total_steps))/2)**(1/2)
        #         #lamda_k=(step_t+1)/total_steps
        #     if 'attn' in name:
        #         r=0.05#0.1,0.05,0.06
        #         #lamda_k=step_t/(r*100+step_t)
        #         #lamda_k=((1-math.cos(math.pi*(step_t+1)/total_steps))/2)**(1/2)
        #         #gamma_buffer.add_(-r*s*grad)
        #         # if sparsity_attn<0.7:
        #         #     z_buffer.add_(-r*step_t/s*grad)
        #         g_sign = (1 - lamda/ torch.abs(z_buffer)).relu()
        #         g_sign = (g_sign * z_buffer) 
        #     if 'mlp' in name:
        #         r=0.125#0.5,0.1,0.2,0.15
                
        #         #gamma_buffer.add_(-r*s*grad)
        #         # if sparsity_mlp<0.7:
        #         #     z_buffer.add_(-r*step_t/s*grad)
        #         g_sign = (1 - lamda/ torch.abs(z_buffer)).relu()
        #         g_sign = (g_sign * z_buffer) 
        #     #g_sign = (param-lamda).relu()
        # #alpha_grad_attn_vision = -((-(g_sign-1)).relu()-1)
        #     #alpha_update = -((-(g_sign-1)).relu()-1)
        #     #param.copy_(lamda_k*g_sign+(1-lamda_k)*gamma_buffer)
        #     param.copy_(gamma_buffer)


def _multi_tensor_adamwprox(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        gamma_buffers: List[Tensor],
                        z_buffers: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        mu: float,
                        kappa: float,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    # update steps
    torch._foreach_add_(state_steps, 1)

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

        torch._foreach_addcdiv_(params, exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sqs = torch._foreach_maximum(max_exp_avg_sqs, exp_avg_sqs)  # type: ignore[assignment]

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)
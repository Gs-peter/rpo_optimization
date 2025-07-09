
import numpy as np
import torch

def getlr( step):

    value  = np.arange(step)
    norm_value = (value - value.min())/ (value.max() - value.min())
    norm_value = norm_value * np.pi
    lr = np.cos( norm_value )

    return lr

increment = 2
total_num_epoch = 100
init_value = 2

epoch_packit = [10]
tracking_total = 10

while tracking_total < total_num_epoch:
    current_epoch = epoch_packit[-1]
    next_epoch = min(
            current_epoch *2,
            abs(current_epoch - total_num_epoch )
            )
    tracking_total += current_epoch *2
    epoch_packit.append(
            next_epoch
            )

num_epoch = 0
for epoch_value in epoch_packit:
    for index , values in getlr(epoch_value):
        logits = model(
                **input_id
                )
        log_prob = F.log_softmax(
                logits
                ).gather(
                        label,
                        dim = -1
                        )

        loss = log_prob.mean()
        loss.beakword()
        for layer_optim in optim.param_groups:
            layer_optim['lr'] = values
        '''
        learning rate setting
        '''
        optim.step()
        num_epoch += 1





~
~
~
~
~
~
~
~
~

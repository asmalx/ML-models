import config, losses, visualization, callbacks
import tensorflow as tf

@tf.function
def train_do_step(gen1, gen2, dis1, dis2, opt_gen1, opt_gen2, opt_dis1, opt_dis2, x1batch, x2batch):

    # learn discriminators
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch([dis1.trainable_variables, dis2.trainable_variables])
        # args: gen1, gen2, dis1, dis2, X1, X2
        dloss = losses.discriminators_loss(gen1, gen2, dis1, dis2, x1batch, x2batch)

    discr_grad1 = tape.gradient(dloss, dis1.trainable_variables)
    discr_grad2 = tape.gradient(dloss, dis2.trainable_variables)
    opt_dis1.apply_gradients(zip(discr_grad1, dis1.trainable_variables))   
    opt_dis2.apply_gradients(zip(discr_grad2, dis2.trainable_variables))   
    del tape

    #  learn generators
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch([gen1.trainable_variables, gen2.trainable_variables])
        # args: gen1, gen2, dis1, dis2, X1, X2
        gloss = losses.generators_loss(gen1, gen2, dis1, dis2, x1batch, x2batch)

    gen_grad1 = tape.gradient(gloss, gen1.trainable_variables)
    gen_grad2 = tape.gradient(gloss, gen2.trainable_variables)
    opt_gen1.apply_gradients(zip(gen_grad1, gen1.trainable_variables))   
    opt_gen2.apply_gradients(zip(gen_grad2, gen2.trainable_variables))   
    del tape
    return gloss, dloss



def fit(data_sequence, gen1, gen2, dis1, dis2, opt_gen1, opt_gen2, opt_dis1, opt_dis2, epochs, callbacks_ = []):
    bar = visualization.ProgressBar()
    history = {'generators_loss':[], 'discriminators_loss':[], 'epoch':[]}
    batches_count = len(data_sequence)

    for eph in range(1, epochs+1):
        avg_gloss, avg_dloss = 0., 0.
        bar.iter_started()
        for batch_idx in range(batches_count):
            # do step
            x1batch, x2batch = data_sequence.__getitem__(batch_idx)
            gloss, dloss = train_do_step(gen1, gen2, dis1, dis2, opt_gen1, opt_gen2, opt_dis1, opt_dis2, x1batch, x2batch)
            
            # recalc AVGs
            avg_gloss = (avg_gloss*batch_idx + gloss.numpy()) / (batch_idx + 1)
            avg_dloss = (avg_dloss*batch_idx + dloss.numpy()) / (batch_idx + 1)
            #  call clbcks
            for c in callbacks_: c.on_batch_end([gen1, gen2, dis1, dis2], [gloss, dloss])
            # draw progress bar
            bar(['Epoch', 'batch'], [epochs, batches_count], [eph, batch_idx+1], [batches_count, batch_idx+1], {'gloss':avg_gloss, 'dloss':avg_dloss})

        # after end of epoch, collect stat, call clbcks
        history['epoch'] += [eph]
        history['generators_loss'] += [avg_gloss]
        history['discriminators_loss'] += [avg_dloss]
        for c in callbacks_: c.on_epoch_end([gen1, gen2, dis1, dis2], [avg_gloss, avg_dloss])
        bar.new_line()
    return history






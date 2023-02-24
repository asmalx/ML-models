import tensorflow as tf
import config
import tensorflow.keras.backend as K



adv_loss_func = tf.keras.losses.MeanSquaredError()
cyc_loss_func = tf.keras.losses.MeanAbsoluteError()
idn_loss_func = tf.keras.losses.MeanSquaredError()

'''
Default GAN discriminator loss, for both models
[dis(fake) vs Zeros. + dis(true) vs Ones.]
'''
@tf.function
def discriminators_loss(gen1, gen2, dis1, dis2, X1, X2):
    # feed generators
    gX2 = gen1(X1)
    gX1 = gen2(X2) 

    # feed discriminators
    dis_real1 = dis1(X1)
    dis_real2 = dis2(X2)
    dis_fake1 = dis1(gX1)
    dis_fake2 = dis2(gX2)

    # default adversarial loss
    dis_real_loss1 = adv_loss_func(dis_real1, tf.ones_like(dis_real1))
    dis_real_loss2 = adv_loss_func(dis_real2, tf.ones_like(dis_real2))   
    dis_fake_loss1 = adv_loss_func(dis_fake1, tf.zeros_like(dis_fake1))
    dis_fake_loss2 = adv_loss_func(dis_fake2, tf.zeros_like(dis_fake2))

    total_loss = dis_real_loss1 + dis_real_loss2 + \
        dis_fake_loss1 + dis_fake_loss2
    
    return total_loss / 2.


'''
Consist of three losses:
    - Default adversarial loss [disY(genX(X)) vs Ones.]
    - Cycle loss [X vs. genY(genX(X))]
    - Identity loss [X vs genY(X)]

gen1, gen2, dis1, dis2 - models
X1, X2 - input data batches
'''
@tf.function
def generators_loss(gen1, gen2, dis1, dis2, X1, X2):
    # feed generators
    gX2 = gen1(X1)
    gX1 = gen2(X2) 

    # default adversarial loss
    discriminator_fake1 = dis1(gX1)
    discriminator_fake2 = dis2(gX2)
    loss_generator1 = adv_loss_func(discriminator_fake1, tf.ones_like(discriminator_fake1))
    loss_generator2 = adv_loss_func(discriminator_fake2, tf.ones_like(discriminator_fake2))

    # cycle consistency loss
    cycle_loss1 = cyc_loss_func(X2, gen1(gX1))
    cycle_loss2 = cyc_loss_func(X1, gen2(gX2))

    # identity loss
    identity_loss1 = idn_loss_func(X2, gen1(X2))
    identity_loss2 = idn_loss_func(X1, gen2(X1))

    # sum up 
    total_loss = loss_generator1 + loss_generator2 + \
        config.CYCLE_LOSS_WEIGHT*(cycle_loss1 + cycle_loss2) + \
        config.IDENTITY_LOSS_WEIGHT*(identity_loss1 + identity_loss2)
   
    return total_loss / 2.

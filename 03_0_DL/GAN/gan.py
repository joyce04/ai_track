import tensorflow as tf
import matplotlib.pyplot as plt
import data
import discriminator, generator
slim = tf.contrib.slim
# import pandas as pd
# import numpy as np

# random_noise is created by random normal distribution for each fake image
# input - batch_size to create random noise equal to the number of batch_size
def random_noise(batch_size, noise):
    return tf.random_normal([batch_size, noise], stddev=0.1).eval()
    # return np.random.normal(loc=0.0, scale=0.01, size=[batch_size, noise])


def input_pipeline(filenames, total_epochs, mini_batch_size):
    filename_queue = tf.train.string_input_producer(filenames)
    real_images = data.read_input_queue(filename_queue)
    label = True
    print(real_images.shape)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    min_after_dequeue = 100
    # capacity recommended by tensorflow
    capacity = min_after_dequeue + mini_batch_size * 3
    image_batch, label_batch = tf.train.shuffle_batch([real_images, label]
                                                      , batch_size=min_after_dequeue
                                                      , capacity=capacity
                                                      , min_after_dequeue=min_after_dequeue)
    return image_batch, label_batch


def run_gan(files, total_epochs, batch_size, model_type):
    n_input = 64 * 64 * 3
    n_noise = 128
    model_path = 'check_points/model_save/'
    mini_batch_size = 50
    d_learning_rate = 2e-4
    g_learning_rate = 2e-3  # 1e-3

    g = tf.Graph()
    with g.as_default():
        train_x, train_y = input_pipeline(files, total_epochs, mini_batch_size)
        # train_x, train_y = mnist.train.next_batch(BATCH_SIZE)

        # noise = random_noise(BATCH_SIZE, n_noise)

        # 1. feed input to graph
        # None because values are wrapped continuously
        X = tf.placeholder(tf.float32, [None, n_input])
        # because GAN is unsupervised learning, it does not require y labels

        # noise
        Z = tf.placeholder(tf.float32, [None, n_noise])

        # 2. generator & discriminator
        if model_type == 'dc':
            print('Training DCGAN....')
            fake_x = generator.dc_generate(Z)

            real_result, real_logists = discriminator.dc_discriminate(X)
            fake_result, fake_logists = discriminator.dc_discriminate(fake_x, True)
        else:
            print('Training Vanilla GAN....')

            fake_x = generator.vanilla_generate(Z)

            real_result, real_logists = discriminator.vanilla_discriminate(X)
            fake_result, fake_logists = discriminator.vanilla_discriminate(fake_x, True)

        # 3. loss functions
        # loss function in GAN represents performance of generator and discriminator
        # both generator and discriminator try to maximize its loss function
        # both want high gen_loss and dis_loss but both are in inverse relationship
        # gen_loss = how fake images are similar to real images
        # dis_loss = how accurate discriminator is to determine which is real and which is fake

        # gen_loss = -tf.reduce_mean(tf.log(fake_result))
        # dis_loss = -tf.reduce_mean(tf.log(real_result) + tf.log(1. - fake_result))
        # one-sided label
        dis_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logists, labels=tf.ones_like(real_logists)))
        dis_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logists, labels=tf.zeros_like(fake_logists)))
        dis_loss = dis_loss_real * 0.7 + dis_loss_fake * 1.0
        # dis_loss = dis_loss_real + dis_loss_fake
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logists, labels=tf.ones_like(fake_logists)))

        # 4. train
        train_vars = tf.trainable_variables()

        gen_vars = [var for var in train_vars if 'gen' in var.name]
        dis_vars = [var for var in train_vars if 'dis' in var.name]

        # tensorflow's optimizer only offers minimization function
        dis_train = tf.train.AdamOptimizer(learning_rate=d_learning_rate, beta1=0.5).minimize(dis_loss,
                                                                                              var_list=dis_vars)
        gen_train = tf.train.AdamOptimizer(learning_rate=g_learning_rate, beta1=0.5).minimize(gen_loss,
                                                                                              var_list=gen_vars,
                                                                                              name='gen_train')

    # iterate training and update variables
    # train_g_loss_his = []
    # train_d_loss_his = []
    with tf.Session(graph=g) as session:
        saver = tf.train.Saver(max_to_keep=3)
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(session, coordinator)
        print('Start training')
        # total_batch = int(mnist.train.num_examples/BATCH_SIZE)
        total_batch = int(batch_size / mini_batch_size)

        for epoch in range(total_epochs):
            for step in range(total_batch):
                # for i in range(total_batch):
                #     train_x, train_y = mnist.train.next_batch(BATCH_SIZE)
                # print(train_x.eval())
                # print(noise)
                # session.run(gen_train, feed_dict={Z: noise})
                # session.run(dis_train, feed_dict={X: train_x.eval(), Z: noise})
                noise = random_noise(batch_size, n_noise)

                _ = session.run(gen_train, feed_dict={Z: noise})
                _ = session.run(dis_train, feed_dict={X: train_x.eval(), Z: noise})
                # _ = session.run(dis_train, feed_dict={X: train_x, Z: noise})

                train_dis_loss = session.run(dis_loss, feed_dict={X: train_x.eval(), Z: noise})
                # train_dis_loss = session.run(dis_loss, feed_dict={X: train_x, Z: noise})
                train_gen_loss = gen_loss.eval({Z: noise})
                # g_loss, d_loss = session.run([gen_loss, dis_loss], feed_dict={X: train_x.eval(), Z: noise})

            # check performance every 10 epoch
            if (epoch + 1) % 50 == 0:
                print("=======Epoch : ", epoch + 1, " =======================================")
                print("generator loss : ", train_dis_loss)
                print("discriminator loss : ", train_gen_loss)
                # train_d_loss_his.append({'x': epoch + 1, 'y': train_gen_loss})
                # train_g_loss_his.append({'x': epoch + 1, 'y': train_dis_loss})

                # check 10 fake images and input images
            if epoch == 0 or (epoch + 1) % 50 == 0:
                # fig, ax = plt.subplots(1, 10, figsize=(10, 1))
                # for i in range(10):
                #     ax[i].set_axis_off()
                #     real_images_ = tf.expand_dims(train_x.eval()[i], 0)
                #     ax[i].imshow(tf.reshape(real_images_, (64, 64, 3)).eval())
                # plt.savefig('check_points/input_{}.png'.format(str(epoch + 1).zfill(3)), bbox_inches='tight')
                # plt.close(fig)

                sample_noise = random_noise(10, n_noise)

                generated = session.run(fake_x, feed_dict={Z: sample_noise})
                fig, ax = plt.subplots(1, 10, figsize=(10, 1))
                for i in range(10):
                    ax[i].set_axis_off()
                    ax[i].imshow(tf.reshape(generated[i], (64, 64, 3)).eval())
                plt.savefig('check_points/{}.png'.format(str(epoch + 1).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                saver.save(session, model_path, global_step=epoch)

        saver.save(session, model_path + '/final')
        print('model saved in file : %s' % model_path)

        coordinator.request_stop()
        coordinator.join(threads)
        print('optimization finished')

    # gen_his = pd.DataFrame(train_g_loss_his)
    # gen_his.columns = ['x', 'y']
    # disc_his = pd.DataFrame(train_d_loss_his)
    # disc_his.columns = ['x', 'y']
    #
    # sub = plt.subplot(2, 1, 1)
    # plt.title('Training loss')
    # plt.xlabel('iteration(Epoch)')
    # plt.plot(gen_his.x.tolist(), gen_his.y.tolist(), '-', label='generator')
    # plt.plot(disc_his.x.tolist(), disc_his.y.tolist(), '-', label='discriminator')
    # plt.legend()
    # plt.gcf().set_size_inches(15, 12)
    # plt.savefig('training_loss.png', bbox_inches='tight')
    # plt.show()


def test_gan(model_type):
    model_path = 'check_points/model_save/'
    n_noise = 128

    z = tf.placeholder(tf.float32, [None, n_noise])

    if model_type == 'dc':
        bn_params = {
            "decay": 0.99,
            "epsilon": 1e-5,
            "scale": True,
            "is_training": True
        }
        initial_shape_multi = 4 * 4 * 1024
        initial_shape = [-1, 4, 4, 1024]

        # to reshape the given noise
        net = z
        net = slim.fully_connected(net, initial_shape_multi, activation_fn=tf.nn.relu)
        net = tf.reshape(net, initial_shape)
        with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5, 5], stride=2, padding='SAME',
                            activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                            normalizer_params=bn_params):
            net = slim.conv2d_transpose(net, 512)
            net = slim.conv2d_transpose(net, 256)
            net = slim.conv2d_transpose(net, 128)
            net = slim.conv2d_transpose(net, 3, activation_fn=tf.nn.tanh, normalizer_fn=None)
            return net
    else:
        n_hidden = 64
        n_noise = 128
        _mean = 0.0
        _stddev = 0.01
        n_output = 64 * 64 * 3

        gw1 = tf.get_variable(name='w1',
                              shape=[n_noise, n_hidden],
                              initializer=generator.xavier_init(n_noise, n_hidden))
        # initializer = tf.random_normal_initializer(mean=_mean, stddev=_stddev))

        gb1 = tf.get_variable(name='b1',
                              shape=[n_hidden],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw2 = tf.get_variable(name='w2',
                              shape=[n_hidden, n_hidden * 2],
                              initializer=generator.xavier_init(n_hidden, n_hidden * 2))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb2 = tf.get_variable(name='b2',
                              shape=[n_hidden * 2],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gw3 = tf.get_variable(name="w3",
                              shape=[n_hidden * 2, n_output],
                              initializer=generator.xavier_init(n_hidden * 2, n_output))
        # initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))
        gb3 = tf.get_variable(name="b3",
                              shape=[n_output],
                              initializer=tf.random_normal_initializer(mean=_mean, stddev=_stddev))

        hidden1 = tf.nn.leaky_relu(tf.matmul(z, gw1) + gb1)
        hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, gw2) + gb2)
        output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)

    saver = tf.train.import_meta_graph(model_path + 'final.meta')
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(model_path))
        session.run(tf.global_variables_initializer())

        Z = random_noise(10, n_noise)

        output_img = session.run(output, feed_dict={'Placeholder:0': Z})
        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(tf.reshape(output_img[i], (64, 64, 3)).eval())
        plt.savefig('check_points/test_{}.png'.format(str(i + 1).zfill(3)), bbox_inches='tight')
        plt.close(fig)

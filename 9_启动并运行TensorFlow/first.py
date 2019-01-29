import time
import tensorflow as tf

if __name__ == "__main__":
    """
    了解tensorflow 基本使用
    """
    # 以下代码创建三个节点并加入默认graph，此时并没有执行任何
    x = tf.Variable(3,name='x')
    y = tf.Variable(4, name='y')
    f = x*x*y + y+ 2

    # 创建会话去执行,result 输出最终值
    # sess  = tf.Session()
    # sess.run(x.initializer)
    # sess.run(y.initializer)
    # result = sess.run(f)
    # sess.close()

    #  上面每次去run太繁琐了，方法块内默认会话,且自动关闭会话
    # with tf.Session() as sess:
    #     x.initializer.run()
    #     y.initializer.run()
    #     result = f.eval()
    # print(result)

    # 上面每个参数都要调一遍初始化太麻烦了,global_variables_initializer,一次性初始化
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     init.run()
    #     result = f.eval()
    # print(result)

    w = tf.constant(3)
    x = w+3
    y = x +5
    z = x*3

    time1 = time.clock()
    with tf.Session() as sess:
        # y,z 在同个运行时，节点值x 被共享不用计算两次
        y_eval,z_eval = sess.run([y,z])
        print(y_eval)
        print(z_eval)
    print("cost time:",time.clock()-time1)

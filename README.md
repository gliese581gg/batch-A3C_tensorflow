#Batch-A3C (bA3C) tensorflow

(Version 1.0, Last updated :2016.12.15)

###1. Introduction

This is modified tensorflow implementation of 'A3C'.(https://arxiv.org/abs/1602.01783)
Instead of using local networks for every threads, this version uses only one global network. Experiences from multiple threads are stored in global experience buffer and used to train the global network. Since the buffer size is small, it does not compromise on-policy learning. There is no significant improvement in performance but this version need less gpu memory and has slightly higher throughput. You can also check my vanilla A3C implementation(https://github.com/gliese581gg/A3C_tensorflow)

I used network composition (layers, activations) of universe starter agent (https://github.com/openai/universe-starter-agent)

![alt tag](https://github.com/gliese581gg/batch-A3C_tensorflow/blob/master/screenshots/bA3C.PNG)





###2. Usage

    python run.py (args)

    where args :

    -log (log directory name) : Tensorboard event file will be crated in 'logs/(your_input)/' (default : 'A3C')
    -ckpt (ckpt file path) : checkpoint file (including path)
    -LSTM (True or False) : whether or not use LSTM layer
    -eval_mode (True or False) : if True, the game is played without learning and score for each episode will be printed
    
    Usage for tensorboard : tensorboard --logdir (your_log_directory) --port (your_port_number)
                            url for tensorboard will appear on terminal:)
                           
    Run with pretrained LSTM agent : python run.py -ckpt ckpt/pre_trained_LSTM -eval_mode True


###3. Requirements:

- Tensorflow
- opencv2
- Arcade Learning Environment ( https://github.com/mgbellemare/Arcade-Learning-Environment )

###4. Test results on 'Pong'
![alt tag](https://github.com/gliese581gg/batch-A3C_tensorflow/blob/master/screenshots/batch-A3CFF.PNG)

Result for Feed-Forward bA3C (took about 1~2 hours, 8 million frames)


![alt tag](https://github.com/gliese581gg/batch-A3C_tensorflow/blob/master/screenshots/batch-A3CLSTM.PNG)

Result for LSTM bA3C (took about 1~2 hours, 3 million frames)


###5. Changelog

-2016.12.15 : First upload!

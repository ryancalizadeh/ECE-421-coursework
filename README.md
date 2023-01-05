# ECE-421-coursework
This repository contains my solutions to 4 assignments for ECE421 Introduction to Machine Learning, taken as part of my degree in Engineering Science. These assignments include implementing and studying Gated Recurrent Neural Networks, Convolution Neural Networks, Support Vector Machines, and the K-means clustering algorithm.

<h1> Gated Recurrent Neural Network </h1>

A gated RNN was trained on samples from the IMBD review dataset to predict whether the overall sentiment is positive or negative. As a preprocessing step, each review was converted into a one-hot-encoded word array and padded to a common length. The one-hot-encoded word arrays were passed through an embedding layer to reduce dimensionality. Then they were passed through a custom Gated Recurrent Unit (GRU), shown below. Finally, they were passed through two fully connected layers to predict a label of 1 (positive sentiment) or 0 (negative sentiment). The network was trained using a custom training routine, and the optimzers Stochastic Gradient Descent and Adam were compared on criteria of efficiency. To conclude the assignment, the early stopping technique was implemented on top of the training routine to investigate it's regularization effect.

```py
class GRU(objax.Module):
    def __init__(self, nin: int, nout: int,
                 init_w: Callable = objax.nn.init.xavier_truncated_normal,
                 init_b: Callable = objax.nn.init.truncated_normal):
        self.update_w = objax.TrainVar(init_w((nin, nout)))
        self.update_u = objax.TrainVar(init_w((nout, nout)))
        self.update_b = objax.TrainVar(init_b((nout,), stddev=0.01))
        self.reset_w = objax.TrainVar(init_w((nin, nout)))
        self.reset_u = objax.TrainVar(init_w((nout, nout)))
        self.reset_b = objax.TrainVar(init_b((nout,), stddev=0.01))
        self.output_w = objax.TrainVar(init_w((nin, nout)))
        self.output_u = objax.TrainVar(init_w((nout, nout)))
        self.output_b = objax.TrainVar(init_b((nout,), stddev=0.01))

    def __call__(self, x: JaxArray, initial_state: JaxArray) -> Tuple[JaxArray, JaxArray]:
        def scan_op(state: JaxArray, x: JaxArray) -> JaxArray:  # State must come first for lax.scan
            update_gate = objax.functional.sigmoid( ( x @ self.update_w ) + ( state @ self.update_u ) + self.update_b )

            reset_gate = objax.functional.sigmoid( ( x @ self.reset_w ) + ( state @ self.reset_u ) + self.reset_b )

            output_gate = objax.functional.tanh( ( x @ self.output_w ) + ( ( reset_gate * state ) @ self.output_u ) + self.output_b )

            return (1-update_gate) * state + (update_gate) * output_gate, 0

        return lax.scan(scan_op, initial_state, x.transpose((1, 0, 2)))[0]
```

<h1> Convolutional Neural Network </h1>

In this assigment, I built and trained a CNN on the CIFAR-10 and CIFAR-100 datasets, and investigated how tuning various hyperparameters affected the training efficiency and performance of the network. Train/Validation accuracy plots were generated and analyzed to determine convergence speed for 7 different CNN's with unique sets of hyperparameters and architectures. For example, below is shown the curve for a CNN with 2 convolutional layers, both having a 3x3 convolution kernel with stride of 2, no padding, with the first outputting 8 channels and the second outputting 16 using global average pooling, and a fully connected layer to complete the network. The model was trained on the CIFAR-10 dataset, using a learning rate of 9e-4, batch size of 32, over 20 epochs with cross-entropy loss.

![image](https://user-images.githubusercontent.com/31375351/210901423-511288d9-bde3-43db-8e65-a866f6d33f3d.png)

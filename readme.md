# Intro

I recently ran across [Weibull Time-to-event Recurrent Neural Networks](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling/ "WTTE-RNN Original Post") (WTTE-RNNs from here on out) for survival prediction. These are the brainchild of Egil Martinsson, a master's degree candidate at the Chalmers University of Technology ([here's his thesis](https://ragulpr.github.io/assets/draft_master_thesis_martinsson_egil_wtte_rnn_2016.pdf "Egil Martinsson Thesis")). Since I do a lot of work with churn data and churn is fundamentally a time-to-event problem, I decided to check them out.

Distilling all of the work in the thesis, the original GitHub post, and the example code down to the bare essentials took quite a bit of doing (or maybe I'm just slow). However, I eventually got my head wrapped around the internals, and decided to code up a bare-bones example using Keras. This is that bare-bones example, trained on some [jet engine failure data from Nasa](https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/ "NASA Prognostics Data Repository").

# The idea

The basic idea of the WTTE-RNN network is this: we want to design a model that can look at a timeline of historical features (jet engine sensor readings, customer behavior, whatever) leading up to the present, and predict a _distribution_ describing the likelihood that a particular event (engine failure, churn) will happen as time moves into the future. If the model is good, it will learn to predict a distribution that is weighted closer to the present for samples that are very close to experiencing an event, and predict a much wider distribution for samples that are unlikely to experience an event any time soon.

If you're a graphical person, we want our model to be able to generate something kind of like this:

![Dummy Model Results](http://i.imgur.com/EXPKvtm.png)

In order to accomplish this, we design a model that predicts the two parameters that control the shape of the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution "Wikipedia: Weibull Distribution"), which we'll call alpha and beta (the literature is all over the place on this, for some reason). The Weibull distribution is commonly used to describe time-to-event data, especially in engineering situations, but we won't go into all that detail here. Suffice it to say, it's a handy distribution for this purpose.

So, we need a neural network that can look at some historical data, and output two parameters describing a distribution that hopefully matches the chances that some event is going to happen to a sample in our data set.

# So, what's the loss function?

In order to train a neural net, you need a loss function that lets you evaluate model performance and backpropagate cost information through the network. In this case, it's not obvious exactly what that should be. And, to make matters worse, there's an additional complication. Time-to-event data is often censored, which means we might know a sample goes at least X time periods without an event, but we haven't observed it long enough to actually find out when the event will happen. Suppose a jet engine has been going strong for 20 years - we know it's been working for 20 years, but we don't know when it'll fail. (If you're new to time-to-event data and censoring, have a look at [my blog post on basic survival analysis](http://daynebatten.com/2015/02/customer-churn-survival-analysis/ "Survival analysis for customer churn") or the [Wikipedia article on survival analysis](https://en.wikipedia.org/wiki/Survival_analysis)).

Thankfully, other people have solved this problem for us. There's a well-known formula for calculating the "log-likelihood" for censored time-to-event data, and folks (including Martinsson) have derived a version specific to the Weibull distribution. Sweet!

I won't go into all the gory math details here, but here's a quick intuitive overview. For all samples (censored or uncensored), the log-likelihood is going to punish our model for predicting high probabilities of events during the known event-free lifetime. So, going back to that engine that's been going strong for 20 years... if our model says it has an 80% chance of having blown up by year 20, the log-likelihood will punish that fairly heavily. It's (most likely) not a good prediction. In addition, for samples where the time of event is known (i.e., not censored) the log-likelihood will reward distributions that give a high chance of experiencing the event _at that point in time_. For censored observations, this part is simply skipped.

If you want more details, you can read Martinsson's thesis, but it ultimately ends up getting implemented this way in Keras:

```python
def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)
```

A few things to note about this... 
* y_true is a (samples, 2) tensor containing time-to-event (y) and a 0/1 event indicator (u).
* ab_pred is a (samples, 2) tensor containing predicted Weibull alpha (a) and beta (b) parameters
* This is the _discrete_ log likelihood function, to be used in situations where your time-to-event data includes discrete time periods (e.g., day 1, day 2, day 3 vs. exact timestamps). The example code includes the continuous log-likelihood as well.
* We're calculating the mean log-likelihood across all samples.
* Usually you want to maximize the log-likelihood, but Keras minimizes loss, so we multiply the whole thing by -1.

So, now we have a function for Keras to minimize. Cool.

# Some Keras tweaks

There are a couple of other bumps in the road to implementing this in Keras that we'll need to navigate. The first of these is that Martinsson recommends using an exponential activation function for alpha and softplus for beta. Unfortunately, Keras doesn't support applying different activation functions to the individual neurons. Thankfully, a custom activation function takes care of this...

```python
def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)
```

The second bump in the road is that Keras doesn't have a super-clean way of implementing char-RNNs (though rumor has it they're working on it). However, François Chollet (the chief contributor to Keras) has [posted a great example of how to accomplish a char-RNN in Keras](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py "LSTM Text Generation Keras"). Essentially, you have to turn each individual time history (say, historical data for one engine) into a _set_ of (almost entirely) overlapping time histories. You'll have one observation per time step (unless you choose to jump a couple steps at a time, as in the linked example), and each observation will contain a history of data leading up to that time step. This results in a tensor of the shape (sample/time step, historical  time steps, features). Check out my code or Chollet's example for an example of how this works.

Of course, this creates an obvious problem: what happens when each time step has a differing amount of data? Say, observation 2 of an engine has 2 observations of history, but observation 100 has 100! To get past this, we simply pad the earlier parts of the time history with zeros, and use a [masking layer in Keras](https://keras.io/layers/core/#masking "Keras masking layer").

# All together now

The actual code to build the Keras model is super-simple. We need a masking layer, an LSTM (RNN) layer, and a dense layer to output 2 neurons. Then, we just need to apply our custom activation function and optimize using our custom log-likelihood loss function. It looks like this:

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_time, 24)))
model.add(LSTM(20, input_dim=24))
model.add(Dense(2))
model.add(Activation(activate))
model.compile(loss=weibull_loglik_discrete, optimizer=RMSprop(lr=.001))
model.fit(train_x, train_y, nb_epoch=250, batch_size=2000, verbose=2, validation_data=(test_x, test_y))
```

# Results

After training this model on the aforementioned jet engine failure data from Nasa (and with no real attempt to optimize anything whatsoever), it becomes apparent that the model is definitely learning _something_. For example, here's what the model predicts for average remaining useful life (survival function = 50%) for each engine in the test data, plotted against the actual remaining useful life:

![Demo WTTE-RNN Performance](http://i.imgur.com/CxCRnzQ.png "Performance Graph for WTTE-RNN")

Many of the engines that the model picks as being closer to failing genuinely are closer to failure, and all of the engines that the model gives a pass to are doing OK. That's not too bad. Of course, there may be a lot more juice to squeeze out here, but I'll leave that as an exercise for somebody else...

# Questions / Comments?

If you've got any questions or comments about what I've posted here, you can obviously open pull requests or issues as appropriate. Also, I've posted a quick blurb about this on my blog: [Recurrent Neural Networks for Churn Prediction](http://daynebatten.com/2017/02/recurrent-neural-networks-churn/ "Churn RNN Blog Post"). I'm usually pretty good about following up on blog comments, so feel free to ask away on there as well.

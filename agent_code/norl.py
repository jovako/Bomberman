#!/usr/bin/env python3

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import pickle

size=4*700000-2*4*70000
#size=200000

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=83, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--new_model", default=0, type=int, help="Need new model?")
# If you add more arguments, ReCodEx will keep them with your default values.
'''
def transformfield(game_state):
    dist=4
    field=-np.ones((2*dist+1,2*dist+1))
    me=game_state["self"][3]
    xmin=max(me[0]-dist,0)
    ymin=max(me[1]-dist,0)
    xmax=min(me[0]+dist+1,17)
    ymax=min(me[1]+dist+1,17)
    fieldxmin=max(dist-me[0],0)
    fieldymin=max(dist-me[1],0)
    fieldxmax=min(17+dist-me[0],2*dist+1)
    fieldymax=min(17+dist-me[1],2*dist+1)
    bombs=game_state["bombs"]
    others=game_state["others"]
    newfield=np.zeros((17,17))
    coins=game_state["coins"]
    for coin in coins:
        newfield[coin]=4
    for other in others:
        newfield[other[3]]=2
    for bomb in bombs:
        newfield[bomb[0]]=-5+bomb[1]
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=(game_state["field"]+newfield)[xmin:xmax,ymin:ymax]
    return field.reshape((2*dist+1)**2)
'''
def transformfield(game_state):
    dist=7
    field=-np.ones((2*dist+1,2*dist+1))
    me=game_state["self"][3]
    xmin=max(me[0]-dist,0)      #magic
    ymin=max(me[1]-dist,0)
    xmax=min(me[0]+dist+1,17)   #more CoOrDs
    ymax=min(me[1]+dist+1,17)
    fieldxmin=max(dist-me[0],0) #random maxmins
    fieldymin=max(dist-me[1],0)
    fieldxmax=min(17+dist-me[0],2*dist+1)
    fieldymax=min(17+dist-me[1],2*dist+1)
    bombs=game_state["bombs"]
    others=game_state["others"]
    newfield=np.zeros((17,17))
    coins=game_state["coins"]
    for coin in coins:     
        newfield[coin]=10
    for other in others:
        newfield[other[3]]=2
    for bomb in bombs:
        newfield[bomb[0]]=-5+bomb[1] #some calculation
    field[fieldxmin:fieldxmax,fieldymin:fieldymax]=(game_state["field"]+newfield)[xmin:xmax,ymin:ymax]      #MoRe InDeXaTiO
    return field.reshape((2*dist+1)**2)

def strtoint(action):
    eye=np.eye(6,dtype=int)
    if action=="UP":
        return eye[0]
    if action=="RIGHT":
        return eye[1]
    if action=="DOWN":
        return eye[2]
    if action=="LEFT":
        return eye[3]
    if action=="WAIT":
        return eye[4]
    if action=="BOMB":
        return eye[5]
    else:
        return eye[4]


def main(args):
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    end=0
    # Load data
    
    file=open(r"t4data.pickle",'rb')
    fields=[]
    labels=[]
    for _ in range(size):
        try:
            o = pickle.load(file)
            f=o[0]
            a=o[1]
            fields.append(f)
            labels.append(a)
        except EOFError:
            end=_
            break

    '''
    
    file=open(r"data5.pickle",'rb')
    fields=[]
    labels=[]
    for _ in range(size):
        try:
            o = pickle.load(file)
            f=transformfield(o[0])
            a=strtoint(o[1])
            fields.append(f)
            labels.append(a)
            #with open(r"t1data.pickle", "ab") as output_file:
                #pickle.dump((f,a), output_file)
        except EOFError:
            end=_
            print(end)
            break
    file.close()
    file=open(r"data2.pickle",'rb')

    for _ in range(size-end):
        try:
            o = pickle.load(file)
            f=transformfield(o[0])
            a=strtoint(o[1])
            fields.append(f)
            labels.append(a)

        except EOFError:
            end+=_
            break
    file.close()
    file=open(r"data3.pickle",'rb')

    for _ in range(size-end):
        try:
            o = pickle.load(file)
            f=transformfield(o[0])
            a=strtoint(o[1])
            fields.append(f)
            labels.append(a)
        except EOFError:
            end+=_
            break

    file.close()
    file=open(r"data4.pickle",'rb')

    for _ in range(size-end):
        try:
            o = pickle.load(file)
            f=transformfield(o[0])
            a=strtoint(o[1])
            fields.append(f)
            labels.append(a)
        except EOFError:
            print("too few data")
            break
    '''
    traindata={"data":np.array(fields),"labels":np.array(labels)}
    fields=[]
    labels=[]



    for _ in range(int(size/10)):
        try:
            o = pickle.load(file)
            f=o[0]
            a=o[1]
            #f=transformfield(o[0])
            #a=strtoint(o[1])

            fields.append(f)
            labels.append(a)
        except EOFError:
            print("not enough data")
            break
    valdata={"data":np.array(fields),"labels":np.array(labels)}

    fields=[]
    labels=[]
    for _ in range(int(size/10)):
        try:
            o = pickle.load(file)
            #f=transformfield(o[0])
            #a=strtoint(o[1])

            f=o[0]
            a=o[1]
            fields.append(f)
            labels.append(a)
        except EOFError:
            print("not enough data")
            break
    testdata={"data":np.array(fields),"labels":np.array(labels)}
    file.close()
    
    if args.new_model == 1: 
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(args.hidden_layer),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(6, activation=tf.nn.softmax),
        ])

        steps = tf.cast(tf.math.ceil(size / args.batch_size), tf.int64)

        if args.decay == None:
            momentum=args.momentum if args.momentum != None else 0.0
            learning_rate=args.learning_rate
        elif args.decay == "polynomial":
            learning_rate = tf.optimizers.schedules.PolynomialDecay(args.learning_rate, steps, end_learning_rate=args.learning_rate_final)
        elif args.decay == "exponential":
            decay_rate = args.learning_rate_final / args.learning_rate
            learning_rate = tf.optimizers.schedules.ExponentialDecay(args.learning_rate, steps, decay_rate, staircase=False)

        if args.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif args.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)



        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
        )
    else:
        model=tf.keras.models.load_model("mymodel")

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    tb_callback._close_writers = lambda: None # Ugly hack allowing to log also test data metrics.
    model.fit(
        traindata["data"],traindata["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(valdata["data"], valdata["labels"]),
        callbacks=[tb_callback],
    )
    model.save("mymodel")
    test_logs = model.evaluate(
        testdata["data"], testdata["labels"], batch_size=args.batch_size, return_dict=True,
    )
    tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

    # Return test accuracy for ReCodEx to validate
    return test_logs["accuracy"]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import EdgeSplitter

import tensorflow as tf
from tensorflow import keras

def trainGCN(graph):
    G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test = create_train_test_sets(graph)
    train_gen, train_flow, test_gen, test_flow = create_train_test_flow(G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test)
    pre_model = create_pre_model(train_gen, test_gen)
    model = train_model(pre_model)
    return model

def create_train_test_sets(G):
    edge_splitter_test = EdgeSplitter(G)
    G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global"
    )
    edge_splitter_train = EdgeSplitter(G_test)
    G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        p=0.3, method="global"
    )
    return G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test

def create_train_test_flow(G_train, edge_ids_train, edge_labels_train, G_test, edge_ids_test, edge_labels_test, batch_size=1024,num_samples=[10,5]):
    train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples, weighted=False)
    train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)
    test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples, weighted=False)
    test_flow = test_gen.flow(edge_ids_test, edge_labels_test)
    return train_gen, train_flow, test_gen, test_flow

def create_pre_model(train_gen, test_gen):
    layer_sizes = [600, 600]
    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, dropout=0.3  #0.2
    )
    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(
        output_dim=1, output_act="relu", edge_embedding_method="ip"
    )(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=["acc"],
    )
    return model

def train_model(model, train_flow, test_flow):
    mcp_save = keras.callbacks.ModelCheckpoint('model_checkpoint.hdf5', verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='min')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-4)
    
    model.fit(train_flow, epochs=5, validation_data=test_flow, verbose=2, use_multiprocessing=True, workers=64, callbacks=[mcp_save, earlystop, reduce_lr])
    
    return model
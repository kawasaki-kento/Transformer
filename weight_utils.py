import pickle
import tensorflow as tf

def save_weights_as_pickle(file_prefix, optimizer, model):
    model_weights = get_model_weights_as_numpy(model)
    optimizer_weights = get_optimizer_weights_as_numpy(optimizer, model)
    all_weights = {'model': model_weights, 'optimizer': optimizer_weights}

    with open(file_prefix + '.pkl', 'wb') as f:
        pickle.dump(all_weights, f)
        
    print('Save checkpoints')

def get_model_weights_as_numpy(model):
    weights = {}
    for v in model.weights:
        weights[v.name] = v.numpy()
    return {'model_name': model.name, 'weights': weights}

def get_optimizer_weights_as_numpy(optimizer, model):
    weights = {}
    slot_names = optimizer.get_slot_names()
    for v in model.weights:
        weights[v.name] = {}
        for slot in slot_names:
            weights[v.name][slot] = optimizer.get_slot(v, slot).numpy()
    return {'optimizer_name': optimizer._name, 'weights': weights}

def load_weights_from_pickle(file_prefix, optimizer, model):
    with open(file_prefix + '.pkl', 'rb') as f:
        weights = pickle.load(f)

    fail_weights = set_model_weights_from_numpy(weights['model']['weights'], model)

    fail_optimizer_weights = set_optimizer_weights_from_numpy(weights['optimizer'], optimizer, model)
    
    if fail_weights != []:
        print('*** Failed to load model weights ***')
        print(fail_weights)
    if fail_optimizer_weights != []:
        print('*** Failed to load optimizer weights ***')
        print(fail_optimizer_weights)        
    if fail_weights == [] and fail_optimizer_weights == []:
        print('Load checkpoints successfully.')
    
def set_model_weights_from_numpy(weights, model):

    fail_weights = []
    for v in model.weights:
        if v.name in weights.keys():
            v.assign(weights[v.name])
        else:
            fail_weights.append(v.name)
            
    return fail_weights

def set_optimizer_weights_from_numpy(weights, optimizer, model):
    
    fail_optimizer_weights = []
    with tf.name_scope(weights['optimizer_name']):
        optimizer_weights = weights['weights']
        
        for v in model.weights:
            if v.name in optimizer_weights.keys():
                for slot in optimizer_weights[v.name].keys():
                    initializer = tf.initializers.Constant(optimizer_weights[v.name][slot])
                    optimizer.add_slot(v, slot, initializer=initializer)
            else:
                fail_optimizer_weights.append(v.name)
                
    return fail_optimizer_weights

import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from dataset.ccl_dataset import get_train_data_by_scores, _get_range_size
def CyclicalTrain(
    model,
    x,
    y,
    data_sizes,
    scores=None,
    batch_size=32,
    epochs=1,
    callbacks=None,
    verbose=1,
    data=None,
):
    total_sample_count = _get_range_size(x, y)
    current_max = 0
    val_accs = []
    train_accs = []
    val_losses = []
    train_losses = []
    result_dict = {}
    for i in range(epochs):
        sample_count_epoch = int(total_sample_count * data_sizes[i])
        sub_x, sub_y = get_train_data_by_scores(
            x, y, sample_count_epoch, scores=scores / scores.sum()
        )
        
        history = model.fit(
            sub_x,
            sub_y,
            epochs=1,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=data,
        )
        current = history.history["val_accuracy"][0]
        current_max = max(current_max, current)
        # "{:.4f}".format(max(self.history["acc"]))
        # print('Current Mx Val Acc', current_max)
        print("Current Max Val Acc", "{:.4f}".format(current_max))
        
        val_accs.append(current)
        train_accs.append(history.history["accuracy"][0])
        val_losses.append(history.history["val_loss"][0])
        train_losses.append(history.history["loss"][0])        
        
        tf.keras.backend.clear_session()
        gc.collect()
    
    result_dict['accuracy'] = train_accs
    result_dict['loss'] = train_losses
    result_dict['val_accuracy'] = val_accs
    result_dict['val_loss'] = val_losses
    
    return model, current_max, result_dict
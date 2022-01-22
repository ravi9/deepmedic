import pickle
import numpy as np
from openvino.inference_engine import IECore

def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice

def main():

    pkl_folder = "/Share/ravi/upenn/data/tcga-deepmedic-batches"
    sub_id = "TCGA-02-0047_1998.12.15_t1_LPS_rSRI_pre"
    batch_i = 10
    feed_dict_ov_to_load = f"{pkl_folder}/{sub_id}-feed_dict_ov-batch_{batch_i}.pkl"
    gt_lbl_ov_to_load = f"{pkl_folder}/{sub_id}-gt_lbl_ov-batch_{batch_i}.pkl"

    with open(feed_dict_ov_to_load, 'rb') as f:
        feeds_dict_ov = pickle.load(f)

    with open(gt_lbl_ov_to_load, 'rb') as f:
        gt_lbl_of_tiles_per_path = pickle.load(f)

    #### Load OpenVINO model
    ov_model_dir = 'examples/tcga/savedmodels/fp32/'
    modelname = 'deepmedic-4-ov.model.ckpt'
    model_xml = f'{ov_model_dir}/{modelname}.xml'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml)
    exec_net = ie.load_network(network=net, device_name="CPU")
    del net


    out_val_of_ops_ov = exec_net.infer(feeds_dict_ov)
    out_val_of_ops = list(out_val_of_ops_ov.values())
    prob_maps_batch = out_val_of_ops[0]

    prob_maps_batch_0 = prob_maps_batch[:,0,:26,:26,:26]
    dice0 = calculate_dice(prob_maps_batch_0, gt_lbl_of_tiles_per_path[1])

    prob_maps_batch_1 = prob_maps_batch[:,1,:26,:26,:26]
    dice1 = calculate_dice(prob_maps_batch_1, gt_lbl_of_tiles_per_path[1])

    print(f"Dice0: {dice0}, Dice1: {dice1}")


if __name__ == "__main__":
    main()
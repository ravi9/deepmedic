import pickle
import numpy as np
import time
from openvino.inference_engine import IECore

def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice

def infer(feeds_dict_ov, gt_lbl_of_tiles_per_path, model_xml, ov_device):
    # Load network to the plugin
    t_load_start = time.time()
    ie = IECore()
    net = ie.read_network(model=model_xml)
    exec_net = ie.load_network(network=net, device_name=ov_device)
    del net
    t_load = time.time() - t_load_start

    t_infer_start = time.time()
    out_val_of_ops_ov = exec_net.infer(feeds_dict_ov)
    t_infer = time.time() - t_infer_start

    out_val_of_ops = list(out_val_of_ops_ov.values())
    prob_maps_batch = out_val_of_ops[0]

    prob_maps_batch_0 = prob_maps_batch[:,0,:26,:26,:26]
    dice0 = calculate_dice(prob_maps_batch_0, gt_lbl_of_tiles_per_path[1])

    prob_maps_batch_1 = prob_maps_batch[:,1,:26,:26,:26]
    dice1 = calculate_dice(prob_maps_batch_1, gt_lbl_of_tiles_per_path[1])

    print(f"Device: {ov_device}, Dice0: {dice0:.4f}, Dice1: {dice1:.4f}, Load time: {t_load:.1f} s, Infer time: {t_infer:.1f} s")

def main():

    pkl_folder = "./"
    sub_id = "TCGA-02-0047_1998.12.15_t1_LPS_rSRI_pre"
    batch_i = 10
    feed_dict_ov_to_load = f"{pkl_folder}/{sub_id}-batch_{batch_i}-feed_dict_ov.pkl"
    gt_lbl_ov_to_load = f"{pkl_folder}/{sub_id}-batch_{batch_i}-gt_lbl_ov.pkl"

    with open(feed_dict_ov_to_load, 'rb') as f:
        feeds_dict_ov = pickle.load(f)

    with open(gt_lbl_ov_to_load, 'rb') as f:
        gt_lbl_of_tiles_per_path = pickle.load(f)

    #### Init OpenVINO model
    ov_model_dir = '../examples/tcga/savedmodels/fp32-ext/'
    modelname = 'deepmedic-4-ov.model.ckpt'
    model_xml = f'{ov_model_dir}/{modelname}.xml'

    ov_device = "CPU"
    infer(feeds_dict_ov, gt_lbl_of_tiles_per_path, model_xml, ov_device)

    ov_device = "GPU"
    infer(feeds_dict_ov, gt_lbl_of_tiles_per_path, model_xml, ov_device)


if __name__ == "__main__":
    main()
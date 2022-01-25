
mo  --input "net/inp_x_test,net/inp_x_sub_0_test" \
--input_shape [10,1,45,45,45],[10,1,26,26,26] \
--input_meta_graph deepmedic-4-ov.model.ckpt.meta \
--output net/transpose_138 \
--output_dir fp32-ext \
--extensions mo_extensions \
--disable_nhwc_to_nchw
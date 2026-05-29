import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

# def lock_memory_on_first_inference(model, input_tensor1,input_tensor2, logger):
#     """
#     Perform a dummy forward pass with feed_data to lock memory for the first inference.
#     This function ensures that a fixed memory allocation is reserved.
#     """
#     logger.info("Locking memory on first inference...")
#     with torch.no_grad():
#         # Prepare the data input in the expected format for feed_data
#         dummy_data = {
#             'lq': input_tensor1,    # Low-quality input
#             'gt': input_tensor1,   # Additional input if needed by feed_data
#             'im_q': input_tensor2
#         }
        
#         # Feed data into the model to set up necessary states
#         model.feed_data_test(dummy_data)
        
#         # Perform inference
#         model.test()
    
#     torch.cuda.synchronize()
#     logger.info("Memory locked for first inference.")
    
def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    
    # # Lock memory on the first inference
    # sample_input1 = torch.randn(1, 3, 2160, 3840).to('cuda')  # Adjust input size as needed
    # sample_input2 = torch.randn(1, 3, 128, 128).to('cuda')
    # lock_memory_on_first_inference(model, sample_input1, sample_input2, logger)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
    
    model.print_final_metrics()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

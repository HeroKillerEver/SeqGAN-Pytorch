import os
import argparse
from solver import Solver
import torch

parser = argparse.ArgumentParser(description='SeqGAN in pytorch', epilog='#' * 75)
parser.add_argument('--gpus', default='', type=str, help='gpu to use: 0, 1, 2, 3, 4 or 0,1,2. Default: cpu')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate. Default: 0.1')
parser.add_argument('--batch', default=64, type=int, help='batch size. Default: 64')
parser.add_argument('--vocab', default=1000, type=int, help='vocabulary size. Default: 1000')
parser.add_argument('--pre_gen_epoch', default=200, type=int, help='num of pre-epochs. Default: 200')
parser.add_argument('--pre_dis_epoch', default=5, type=int, help='num of pre-epochs. Default: 5')
parser.add_argument('--gan_epoch', default=100, type=int, help='num of gan-epochs. Default: 100')
parser.add_argument('--generate_num', default=10000, type=int, help='num of generated samples. Default: 10000')
parser.add_argument('--sequence_len', default=20, type=int, help='length of sequence. Default: 20')
parser.add_argument('--update_rate', default=0.8, type=float, help='update rate for rollout policy. Default: 0.8')
args = parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
real_data = 'real.csv'
fake_data = 'fake.csv'
eval_data = 'eval.csv'



def main():
	
	if not os.path.isdir('data'):
		os.makedirs('data')
	real_file = os.path.join('data', real_data)
	fake_file = os.path.join('data', fake_data)
	eval_file = os.path.join('data', eval_data)


	solver = Solver(args.vocab, args.batch, args.pre_gen_epoch, args.pre_dis_epoch, 
					args.gan_epoch, args.generate_num, args.sequence_len, args.lr, 
					real_file, fake_file, eval_file, args.update_rate)



	if args.gpus == '':
		print 'SeqGAN in cpu......'
	else:
		print 'SeqGAN in gpu: {}'.format(args.gpus)

	backend = 'cpu' if args.gpus == '' else 'gpu'


	solver.pretrain_gen()
	solver.pretrain_dis()
	solver.train_gan(backend)





if __name__ == '__main__':
	main()
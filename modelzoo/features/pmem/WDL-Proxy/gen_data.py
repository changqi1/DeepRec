import argparse
import os

def gen_fake_data(file,num):
    if num<0:
        print("Error.Num shoule be a positive integer.")
        return

    with open(file,'w') as f:
        for i in range(num):
            line='1'
            for ii in range(39):
                if(ii == 9 ):
                     #line ='%d' % (ii + i%2)
                     line+=',%d' % (ii + i%2)
                else:
                     line+=',%d' % i
            line+='\n'
            f.write(line)
        pass

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        help='Full path of output',
                        default='./')
    parser.add_argument('--num',
                        help='number of fake data.',
                        required=True,
                        type=int)
    parser.add_argument('--name',
                        help='name of fake data.',
                        type=str,
                        default='train.csv')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    dataset_path = os.path.join(args.output_dir, args.name)
    gen_fake_data(dataset_path,args.num)


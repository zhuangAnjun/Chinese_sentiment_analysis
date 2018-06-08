import chardet
from pandas import read_csv

symbols = ['*', '，','。','%',')','）','(','（','》','《','/','&',':',';','-','“','”','"','@','、','{','}','【','】','［','］','：','／','；','－']
nums = ['0','1','2','3','4','5','6','7','8','9']

# 用来处理训练数据还有测试数据
def process_data(input_file, output_file):
    num = 0
    f = read_csv(input_file, delimiter='\t', header=None, encoding='gbk')
    with open(output_file,'w') as file:
        for line in f[2]:
            num += 1
            line = str(line)
            line.encode('utf-8')
            content = ''
            for c in line:
                if c not in symbols and c != '　' and c!='\t':
                    content += c
                    content += ' '
                    
            file.write(content+'\n')
            if num%1000==0:
                print('already deal '+str(num)+' rows')

# 处理搜狗语料库
def process_corpus(input_file, output_file):
    num = 0
    with open(input_file,'w',encoding='utf-8') as outfile:
        with open(output_file,'r',encoding='GB18030') as infile:
            line = infile.readline()
            while line:
                if line[0:9] == '<content>':
                    line = line[9:-11]
                    line = str(line)
                    line.encode('utf-8')
                    content = ''
                    for c in line:
                        if c not in symbols and c != '　':
                            content += c
                            content += ' '
                        if c == '。' or c=='；':
                            outfile.write(content+'\n')
                            content = ''
                line=infile.readline()
                num += 1

                if num%10000 == 0:
                    print('already deal '+str(num)+' rows')
        #                 print(line)

train_data = open('train.data', 'r')
dev_data = open('dev.data', 'r')
test_data = open('test.data', 'r')

train_pad = open('train_pad.data', 'w')
dev_pad = open('dev_pad.data', 'w')
test_pad = open('test_pad.data', 'w')


pad_element = ''
for x in range(20):
    if x == 19:
        pad_element += '0'
    else:
        pad_element += '0 '

for line in train_data:
    line = line.strip()

    line = line.split('\t\t')
    x_data = line[2:len(line)-1]
    while(len(x_data)<16):
        x_data.insert(0, pad_element)


    x_data.insert(0, line[1])
    x_data.insert(0, line[0])
    x_data.append(line[len(line)-1])

    for x in range(len(x_data)):
        if x == len(x_data)-1:
            train_pad.write(x_data[x]+'\n')
        else:
            train_pad.write(x_data[x]+'\t\t')

for line in dev_data:
    line = line.strip()

    line = line.split('\t\t')
    x_data = line[2:len(line)-1]
    while(len(x_data)<16):
        x_data.insert(0, pad_element)


    x_data.insert(0, line[1])
    x_data.insert(0, line[0])
    x_data.append(line[len(line)-1])

    for x in range(len(x_data)):
        if x == len(x_data)-1:
            dev_pad.write(x_data[x]+'\n')
        else:
            dev_pad.write(x_data[x]+'\t\t')

for line in test_data:
    line = line.strip()

    line = line.split('\t\t')
    x_data = line[2:len(line)-1]
    while(len(x_data)<16):
        x_data.insert(0, pad_element)


    x_data.insert(0, line[1])
    x_data.insert(0, line[0])
    x_data.append(line[len(line)-1])

    for x in range(len(x_data)):
        if x == len(x_data)-1:
            test_pad.write(x_data[x]+'\n')
        else:
            test_pad.write(x_data[x]+'\t\t')



train_data.close()
dev_data.close()
test_data.close()

train_pad.close()
dev_pad.close()
test_pad.close()

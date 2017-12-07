
data = open('output', 'r')
train = open('train.data', 'w')
dev = open('dev.data', 'w')
test = open('test.data', 'w')


count = 1

train_threshold = 2738
dev_threshold = 3651

for line in data:
    line = line.strip()
    if line:
        if count <= train_threshold:
            train.write(line+'\n')
        elif count > train_threshold and count<=dev_threshold:
            dev.write(line+'\n')
        elif count > dev_threshold:
            test.write(line+'\n')
        count += 1

data.close()
train.close()
test.close()
dev.close()

data = open('words_final.txt', 'r')
output = open('data.txt', 'w')

EZ = '0'
S = '1'
Z = '2'

sibilants = ['s', 'z', 'S', 'Z', 'tS', 'dZ']
voiceless_cons = ['p', 't', 'k', 'f', 'T']

for line in data:
    line = line.strip()
    if line:
        line = line.split('\t')
        item = line[2]
        last_sound = item[len(item)-1]

        if last_sound in sibilants:
            line.append(EZ)

        elif last_sound in voiceless_cons:
            line.append(S)

        else:
            line.append(Z)

        output.write(line[0]+'\t\t'+line[2]+'\t\t'+line[3]+'\n')

data.close()
output.close()

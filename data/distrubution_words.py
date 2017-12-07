
words = open('words.txt', 'r')

dist = {}
for x in range(1, 17):
    dist[x] = 0

for word in words:
    word = word.strip()
    if not word:
        continue 
    size = len(word)
    if size == 16:
        print word
    dist[size] += 1

print dist

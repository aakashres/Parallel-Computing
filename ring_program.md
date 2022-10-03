magic number = 42
rank = getmyrank()
count = howmanyarethre()

if rank == 0 and count > 1:
    send(magic number, 1)
    recv(magic number, count - 1)

for(i =1 i < count i++){
    recv(magic number, i-1)
    send(magic number, (i + 1)% count)
}
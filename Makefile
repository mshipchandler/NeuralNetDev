# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11 -Wall

# Directories
OBJDIR = ./obj
BINDIR = ./bin

default: clean directories binary
	@echo "Binary created inside the ./bin directory."

directories:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)

# .out extension (runNet.out) for the .gitignore to ignore the binary
binary: Node.o setandrun.o
	@$(CC) $(CFLAGS) -o $(BINDIR)/runNet.out $(OBJDIR)/Node.o $(OBJDIR)/setandrun.o

Node.o: ./src/Node.cpp ./src/Node.h ./src/activation_functions.h
	@$(CC) $(CFLAGS) -c ./src/Node.cpp -o ./obj/Node.o

setandrun.o: ./src/setandrun.cpp ./src/Node.h ./src/data.h ./src/activation_functions.h
	@$(CC) $(CFLAGS) -c ./src/setandrun.cpp -o ./obj/setandrun.o

clean:
	@-rm -rf $(OBJDIR) $(BINDIR)	
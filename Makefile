# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11 -Wall -fopenmp

# OpenCV flags and libraries
OPENCV = `pkg-config --cflags --libs opencv`

# Directories
OBJDIR = ./obj
BINDIR = ./bin

default: directories binary
	@echo "Binary created inside the ./bin directory."

image_preprocessor: directories image_preprocessor_binary
	@echo "Binary created inside the ./bin directory."

descriptor_extractor: directories descriptor_extractor_binary
	@echo "Binary created inside the ./bin directory."

directories:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)

# .out extension (runNet.out) for the .gitignore to ignore the binary
binary: Node.o setandrun.o
	@$(CC) $(CFLAGS) -o $(BINDIR)/runNet.out $(OBJDIR)/Node.o $(OBJDIR)/setandrun.o

# .out extension (runNet.out) for the .gitignore to ignore the binary
image_preprocessor_binary: Node.o setandrun_withImagePreprocessor.o
	@$(CC) $(CFLAGS) -o $(BINDIR)/runNet.out $(OBJDIR)/Node.o $(OBJDIR)/setandrun_withImagePreprocessor.o $(OPENCV)

descriptor_extractor_binary: Node.o setandrun_withDescriptorExtraction.o
	@$(CC) $(CFLAGS) -o $(BINDIR)/runNet.out $(OBJDIR)/Node.o $(OBJDIR)/setandrun_withDescriptorExtraction.o $(OPENCV)

Node.o: ./src/Node.cpp ./src/Node.h ./src/activation_functions.h
	@$(CC) $(CFLAGS) -c ./src/Node.cpp -o ./obj/Node.o

setandrun.o: ./src/setandrun.cpp ./src/Node.h ./src/data.h ./src/activation_functions.h
	@$(CC) $(CFLAGS) -c ./src/setandrun.cpp -o ./obj/setandrun.o

setandrun_withImagePreprocessor.o: ./src/setandrun_withImagePreprocessor.cpp ./src/Node.h ./src/data.h ./src/activation_functions.h ./src/image_preprocessor.h
	@$(CC) $(CFLAGS) -c ./src/setandrun_withImagePreprocessor.cpp -o ./obj/setandrun_withImagePreprocessor.o

setandrun_withDescriptorExtraction.o: ./src/setandrun_withDescriptorExtraction.cpp ./src/Node.h ./src/data.h ./src/activation_functions.h
	@$(CC) $(CFLAGS) -c ./src/setandrun_withDescriptorExtraction.cpp -o ./obj/setandrun_withDescriptorExtraction.o

clean:
	@-rm -rf $(OBJDIR) $(BINDIR)
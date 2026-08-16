CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
SRCDIR = .
SOURCES = symnmf.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTABLE = symnmf

# Default target
all: $(EXECUTABLE)

# Build the executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTABLE) -lm

# Compile source files to object files
%.o: %.c symnmf.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Build Python extension (optional target)
python_ext:
	python3 setup.py build_ext --inplace

# Clean Python extension build artifacts
clean_python:
	rm -rf build/
	rm -f *.so
	rm -f symnmf.c.o
	rm -f symnmfmodule.c.o

# Full clean
clean_all: clean clean_python

.PHONY: all clean python_ext clean_python clean_all
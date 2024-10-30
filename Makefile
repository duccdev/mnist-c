CC := $(or $(shell echo $$CC),clang)
CFLAGS := -o bin/mnist-c -Isrc -O3 -Wall -Wextra -lm $(shell echo $$CFLAGS)

all:
	$(CC) $(CFLAGS) -s src/*.c

debug:
	$(CC) $(CFLAGS) -g src/*.c

format:
	clang-format -style=llvm -i src/*.c

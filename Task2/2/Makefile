# Компилятор
CPP = gcc
# Флаги компиляции
CPPFLAGS = -std=c99 -g -Wall -O2 -fopenmp

# Директория для выходных файлов
OUT_DIR = out

# Количество потоков
THREADS_COUNTS = 1 2 4 7 8 16 20 40

# Генерация всех возможных комбинаций
TARGETS = $(foreach threads,$(THREADS_COUNTS),$(OUT_DIR)/TASK_T$(threads))

# Правило по умолчанию
all: $(TARGETS)

# Правило для компиляции каждого варианта
$(OUT_DIR)/TASK_%:
	$(eval FILENAME=$(notdir $@))
	$(eval MATRIX_SIZE=$(word 1,$(subst _, ,$(subst TASK_,,$(FILENAME)))))
	$(eval THREADS_CNT=$(subst T,,$(lastword $(subst _, ,$(FILENAME)))))
	$(CPP) $(CPPFLAGS) -DTHREADS_CNT=$(THREADS_CNT) -o $@ task.c -lm

# Создание директории для выходных файлов
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Очистка
clean:
	rm -rf $(OUT_DIR)

# Убедимся, что директория out создана перед компиляцией
$(TARGETS): | $(OUT_DIR)

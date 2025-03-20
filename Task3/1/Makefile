# Компилятор
CXX = g++
# Флаги компиляции
CXXFLAGS = -std=c++20 -Wall -O2

# Директория для выходных файлов
OUT_DIR = out

# Количество потоков
THREADS_COUNTS = 1 2 4 7 8 16 20 40

# Размеры матриц
MATRIX_SIZES = 20000x20000 40000x40000

# Генерация всех возможных комбинаций
TARGETS = $(foreach size,$(MATRIX_SIZES),$(foreach threads,$(THREADS_COUNTS),$(OUT_DIR)/TASK_$(size)_T$(threads)))

# Правило по умолчанию
all: $(TARGETS)

# Правило для компиляции каждого варианта
$(OUT_DIR)/TASK_%:
	$(eval FILENAME=$(notdir $@))
	$(eval MATRIX_SIZE=$(word 1,$(subst _, ,$(subst TASK_,,$(FILENAME)))))
	$(eval MATRIX_N=$(firstword $(subst x, ,$(MATRIX_SIZE))))
	$(eval MATRIX_M=$(lastword $(subst x, ,$(MATRIX_SIZE))))
	$(eval THREADS_CNT=$(subst T,,$(lastword $(subst _, ,$(FILENAME)))))
	$(CXX) $(CXXFLAGS) -DTHREADS_CNT=$(THREADS_CNT) -DMATRIX_N=$(MATRIX_N) -DMATRIX_M=$(MATRIX_M) -o $@ task.cpp

# Создание директории для выходных файлов
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Очистка
clean:
	rm -rf $(OUT_DIR)

# Убедимся, что директория out создана перед компиляцией
$(TARGETS): | $(OUT_DIR)

# Компилятор
CXX = pgc++
# Флаги компиляции
CXXFLAGS = -acc -Minfo=all -std=c++11 -Wall -O2

# Директория для выходных файлов
OUT_DIR = out

# Размеры матриц
MATRIX_SIZES = 128x128 256x256 512x512 1024x1024

# Генерация всех возможных комбинаций
TARGETS = $(foreach size,$(MATRIX_SIZES),$(OUT_DIR)/TASK_$(size))

# Правило по умолчанию
all: $(TARGETS)

# Правило для компиляции каждого варианта
$(OUT_DIR)/TASK_%:
	$(eval FILENAME=$(notdir $@))
	$(eval MATRIX_SIZE=$(word 1,$(subst _, ,$(subst TASK_,,$(FILENAME)))))
	$(eval MATRIX_N=$(firstword $(subst x, ,$(MATRIX_SIZE))))
	$(eval MATRIX_M=$(lastword $(subst x, ,$(MATRIX_SIZE))))
	$(CXX) $(CXXFLAGS) -acc=host -DNX=$(MATRIX_N) -DNY=$(MATRIX_M) -o $@_H therm_acc1.cpp
	$(CXX) $(CXXFLAGS) -acc=multicore -DNX=$(MATRIX_N) -DNY=$(MATRIX_M) -o $@_M therm_acc1.cpp
	$(CXX) $(CXXFLAGS) -acc=gpu -DNX=$(MATRIX_N) -DNY=$(MATRIX_M) -o $@_G therm_acc1.cpp

# Создание директории для выходных файлов
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Очистка
clean:
	rm -rf $(OUT_DIR)

# Убедимся, что директория out создана перед компиляцией
$(TARGETS): | $(OUT_DIR)

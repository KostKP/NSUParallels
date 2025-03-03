# Компилятор
CXX = g++

# Флаги компиляции
CXXFLAGS = -Wall -O2 -fopenmp

# Директория для выходных файлов
OUT_DIR = out

# Генерация всех файлов
TARGETS = $(OUT_DIR)/TASK0 $(OUT_DIR)/TASK1 $(OUT_DIR)/TASK2

# Правило по умолчанию
all: $(TARGETS)

# Правило для компиляции каждого варианта
$(OUT_DIR)/TASK0: task0.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(OUT_DIR)/TASK1: task1.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

$(OUT_DIR)/TASK2: task2.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static" -o $@-S $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic" -o $@-D $< -lm

	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 1" -o $@-S1 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 2" -o $@-S2 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 4" -o $@-S4 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 8" -o $@-S8 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 16" -o $@-S16 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 32" -o $@-S32 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 64" -o $@-S64 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 128" -o $@-S128 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="static, 256" -o $@-S256 $< -lm

	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 1" -o $@-D1 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 2" -o $@-D2 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 4" -o $@-D4 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 8" -o $@-D8 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 16" -o $@-D16 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 32" -o $@-D32 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 64" -o $@-D64 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 128" -o $@-D128 $< -lm
	$(CXX) $(CXXFLAGS) -DSCHEDULE="dynamic, 256" -o $@-D256 $< -lm

# Создание директории для выходных файлов
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Очистка
clean:
	rm -rf $(OUT_DIR)

# Убедимся, что директория out создана перед компиляцией
$(TARGETS): | $(OUT_DIR)

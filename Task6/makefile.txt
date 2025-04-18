HEIGHT = 20
WIDTH = 30

default: show_image

.PHONY:
show_image: result.dat
	gnuplot -e "plot 'result.dat' binary \
		array=($(WIDTH), $(HEIGHT)) format='%lf' with image; pause -1"

result.dat: solver
	./solver

solver: therm.c
	gcc -o solver therm.c -lm -DNY=$(HEIGHT) -DNX=$(WIDTH)

clean:
	rm -f solver result.dat

import sys
import csv
import re

def extract_and_save_data(input, mode, output_name):
    if mode not in {'Full', 'IterativeSearch'}:
        print("Error: Second argument must be 'Full' or 'IterativeSearch'.")
        return

    data = []

    if mode == 'Full':
        patron = re.compile(
            r'Time for algorithm:\s*([\d.]+).*?Depth:\s*(\d+).*?Reeval_level:\s*(\d+)',
            re.IGNORECASE
        )

        with open(input, 'r', encoding='utf-8') as f:
            for linea in f:
                if 'Time for algorithm' in linea:
                    match = patron.search(linea)
                    if match:
                        tiempo = float(match.group(1))
                        profundidad = int(match.group(2))
                        reeval = int(match.group(3))
                        data.append([profundidad, reeval, tiempo])

        # Cabecera simple
        headers = ['Depth', 'Reeval_level', 'Time(s)']

    elif mode == 'IterativeSearch':
        patron_modelo = re.compile(
            r'Selected model: depth (\d+) with n_preevals \d+ and method (\w+)\s+with value ([\d.]+)',
            re.IGNORECASE
        )
        patron_tiempo = re.compile(
            r'Time for algorithm:\s*([\d.]+).*?Depth:\s*(\d+).*?Reeval_level:\s*(\d+)',
            re.IGNORECASE
        )

        modelo_actual = {
            'N_best': None,
            'percentage': None,
            'percentage_epsilon': None
        }

        with open(input, 'r', encoding='utf-8') as f:
            for linea in f:
                match_modelo = patron_modelo.search(linea)
                if match_modelo:
                    metodo = match_modelo.group(2)
                    valor = match_modelo.group(3)

                    modelo_actual = {
                        'N_best': None,
                        'percentage': None,
                        'percentage_epsilon': None
                    }

                    if metodo == 'N_best':
                        modelo_actual['N_best'] = int(valor)
                    elif metodo == 'percentage':
                        modelo_actual['percentage'] = float(valor)
                    elif metodo == 'percentage_epsilon':
                        modelo_actual['percentage_epsilon'] = float(valor)
                    continue

                match_tiempo = patron_tiempo.search(linea)
                if match_tiempo:
                    tiempo = float(match_tiempo.group(1))
                    profundidad = int(match_tiempo.group(2))
                    reeval = int(match_tiempo.group(3))

                    data.append([
                        profundidad,
                        reeval,
                        modelo_actual['N_best'],
                        modelo_actual['percentage'],
                        modelo_actual['percentage_epsilon'],
                        tiempo
                    ])

        # Cabecera extendida
        headers = ['Depth', 'Reeval_level', 'N_best', 'percentage', 'percentage_epsilon', 'Time(s)']

    if not data:
        print("Data not found.")
        return

    with open(output_name, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"Saved data to {output_name}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Use: python extract_info_for_slurms.py <fichero.txt> <Completo|IterativeSearch> <output_name>")
    else:
        input = sys.argv[1]
        mode = sys.argv[2]
        output_name = sys.argv[3]
        extract_and_save_data(input, mode,output_name)

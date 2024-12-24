#%% Consignas Clase I

"""
    Nombre: Delitos CABA

    Descripción: Información de delitos para todo el país. Acá sólo usaremos los datos de la Ciudad de Buenos Aires para 2022

    Preguntas:

        Clase I

            ¿Cuál es el crimen más cometido a lo largo de los años? ¿En qué año se presenció la mayor cantidad de veces?

            ¿A qué refiere la tasa de hechos de un delito? ¿Y la de víctimas?

            ¿El año de menor tasa de homicidios dolosos coincide con el año de menor tasa de víctimas de este crimen?

            ¿En qué año se registraron más crímenes en total? ¿Y menos?

            Viendo por ejemplo las tasas de víctimas para el año 2019, en qué tipo de delitos la tasa de víctimas femeninas supera a la tasa masculina?

            ¿Cuál es el delito con la mayor cantidad de víctimas sin definir su género (sd)?

            Generar un dataframe donde, para cada delito, se obtenga el promedio de la cantidad de hechos a lo largo de los años, con su desvío estándar correspondiente.

            Agregar una columna con el año de más prevalencia de cada delito, con la cantidad de veces que sucedió y cantidad de víctimas.
"""

#%% Inicio, análisis previo para visualizacion rapida de los datos.

import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import numpy as np

df = pd.read_csv("delitos_CABA.csv", sep = ";",encoding= "latin1")

df.head(10)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe)

#%% Limpieza de datos

"""
    Analizando los datos de las tasas notamos que están escritos con "," lo que imposibilita que python las lea como 
    números, para ello tendremos que depurar la tabla y transformar estas columnas en float.
"""

df["tasa_hechos"] = df["tasa_hechos"].str.replace(",", ".")
df["tasa_victimas"] = df["tasa_victimas"].str.replace(",", ".")
df["tasa_victimas_masc"] = df["tasa_victimas_masc"].str.replace(",", ".")
df["tasa_victimas_fem"] = df["tasa_victimas_fem"].str.replace(",", ".")

df["tasa_hechos"] = df["tasa_hechos"].astype(float)
df["tasa_victimas"] = df["tasa_victimas"].astype(float)
df["tasa_victimas_masc"] = df["tasa_victimas_masc"].astype(float)
df["tasa_victimas_fem"] = df["tasa_victimas_fem"].astype(float)


#%% 1_ ¿Cuál es el crimen más cometido a lo largo de los años? ¿En qué año se presenció la mayor cantidad de veces?

#Función que muestre el crimen más cometido sumando todos los años del dataset.
def mas_cometido(df):
    
    #Igualamos la var al df agrupado en los tipos de delito, sumando la cantidad de hecho
    #y devolviendo el índice del máximo de esa suma que será el nombre del delito.
    delito = df.groupby(["codigo_delito_snic_nombre"])["cantidad_hechos"].sum().idxmax()
    
    #Retornamos la variable.
    return delito

#Función que devuelve el año en donde el crimen que introducis como parámetro se cometió más veces.
def año_aparicion(df, delito): 
    
    #Selecciona todas las filas de las columnas, "anio", "codigo_delito_snic_nombre", "cantidad_hechos".
    delitos_años = df.iloc[:, 1:4]
    
    #Reduce la selección a solo las filas con el tipo de delitos a analizar.
    el_delito = delitos_años[delitos_años["codigo_delito_snic_nombre"] == delito] 
    
    #igualamos la variable al índice donde se encuentra la fila del año de mayo ocurrencia del delito.
    indice = el_delito.cantidad_hechos.idxmax() 
    
    #Iguala la variable  con el año de mayo cantidad de hechos del delito.
    año = el_delito.loc[indice, "anio"]
    
    #Iguala la variable a la cantidad de hechos registrados.
    cuanto = el_delito.cantidad_hechos.max()
    
    #Retorna las dos variables en forma de tupla.
    return año,cuanto

"""
Prueba de las funciones anteriores.

mas_cometido = mas_cometido(df)
año,cuanto = año_aparicion(df, mas_cometido)

"""

#%% 2_ ¿A qué refiere la tasa de hechos de un delito? ¿Y la de víctimas?

"""
    Deducción: Supusimos que la tasa debía ser una división de una cantidad (hechos o víctimas) entre la población
    multiplicada por alguna potencia de 10, probamos hasta la 5ta. Despejando, llegamos a la hipótesis
    de que la población al momento de los datos= cantidad(H o V)/tasa conocida * 10**5. Utilizaremos la primer función para 
    aproximar la población de este modo y ver si es razonable. Luego, chequear a partir de esa población si podemos estimar la tasa
    real con algún margen de error aceptable. El único fin de estas funciones es comprobar la deducción.
"""
 
def aproximar_poblacion(df, anio, delito, cantidad, tasa):
    """
    Devuelve población estimada según un anio de datos, cantidad (H o V) y una tasa (H o V)
    PRECONDICIONES: -Existe el df y los parámetros deben existir en el Data Frame como columnas.
                    -La tasa real (año[tasa]) debe ser distinta de cero y nan.
    """
    #Filto mi selección
    seleccion = df[(df['anio'] == anio) & (df['codigo_delito_snic_nombre'] == delito)]
    
    #Estimo poblacion dividiendo la cantidad (H o V) sobre la tasa real (H o V) por un factor 10**5.
    poblacion_estimada= (seleccion[cantidad] / seleccion[tasa]) * 10**5
    
    return poblacion_estimada

def aproximar_tasa(df,anio, delito, cantidad, tasa):
    
    """
    Devuelve  la tasa aproximada segun un año de datos, cantidad (H o V) y una tasa (H o V),
    Utilizando la población aproximada. 
    PRECONDICIONES: -Existe el df y los parámetros deben existir en el Data Frame como columnas y ser iguales 
                     a los utilizados para calcular la función anterior. "población" debe ser distinta de 0. 
    """
    
    seleccion= df[(df['anio'] == anio) & (df['codigo_delito_snic_nombre'] == delito)]
    poblacion= aproximar_poblacion(df, anio, delito, cantidad, tasa)
    tasa_estimada=(seleccion[cantidad] / poblacion) * 10**5
    
    return tasa_estimada

def comparacion (df, anio, delito, tasa, tasa_estimada):
    
    """
    Devuelve true o false dependiendo de si la diferencia entre tasa real y estimada es menor
    o igual a una tolerancia establecida. 
    PRECONDICIONES: -Tasa_estimada es una serie de una sola fila resultante de la función anterior y la tasa 
                     ingresada es de la misma característica que la utilizada para calcular la estimación.
    """
    iguales= False
    
    #Hago un filtro seleccionando la parte que me interesa del dataframe.
    seleccion= df[(df['anio'] == anio) & (df['codigo_delito_snic_nombre'] == delito)]
    
    #Establezco una tolerancia igual o menor al primer quartil de 1. Es  un error estimado. 
    tolerancia= 0.25
    
    #Si la selección en los parámetros dados está vacía, no se puede calcular la comparación
    #utilizó el método .empty para cerciorarme de ello.
    if not seleccion.empty:
        
        #Definir las tasas de modo tal que sean de tipo float (no series).
        #El valor de verdad de una serie es ambiguo.
        tasa_real= seleccion.loc[0,tasa]
        tasa_estimada= tasa_estimada.iloc[0]
        
        #Defino la diferencia utilizando el método abs para valor absoluto.
        diferencia= abs(tasa_estimada - tasa_real)
        
        #Si la diferencia es menor o igual que la tolerancia.
        if diferencia <= tolerancia: 
            #Entonces podemos decir que son iguales.
            iguales = True 
        
    else: 
        #Si no lo son, entonces no son iguales y el método usado está fallando. 
        iguales= False
    
    return iguales


'''
Prueba de las funciones anteriores

#Población estimada del 2000 = 2.994.974
poblacion2000= aproximar_poblacion(df, 2000, "Homicidios dolosos", "cantidad_hechos", "tasa_hechos")

#Tasa estimada = 4.975001
tasa_estimada= aproximar_tasa(df, 2000, "Homicidios dolosos", "cantidad_hechos", "tasa_hechos")

seleccion= df[(df['anio'] == 2000) & (df['codigo_delito_snic_nombre'] == "Homicidios dolosos")]

#True
verdad= comparacion(df, 2000, "Homicidios dolosos", "tasa_hechos", tasa_estimada)

#Tasa real = 4.9750013
tasa_real = seleccion.loc[0, "tasa_hechos"]

'''

#%% 3_ ¿El año de menor tasa de homicidios dolosos coincide con el año de menor tasa de víctimas de este crimen?

#Insertada un tipo de tasa y delito devuelve el año y la tasa mínima.
def menor_tasa(tasa, delito, df):
    
    #Dataset menor con sólo los crímenes del delito que se busca.
    por_delito = df[df.codigo_delito_snic_nombre == delito]
    
    #Agrupa por año, suma los resultados de la tasa que se está analizando y devuelve el índice, o sea el año, de la menor.
    año = por_delito.groupby(["anio"])[tasa].sum().idxmin()
    
    #Devuelvo la menor tasa.
    tasa = por_delito[tasa].min()
    
    #Retorna las dos variables en forma de tupla.
    return año , tasa

#Insertada un tipo de tasa y delito devuelve el año y la tasa máxima.
def mayor_tasa(tasa, delito, df):
    
    #Dataset menor con sólo los crímenes del delito que se busca.
    por_delito = df[df.codigo_delito_snic_nombre == delito]
    
    #Agrupa por año, suma los resultados de la tasa que se está analizando y devuelve el índice, o sea el año, de la mayor.
    año = por_delito.groupby(["anio"])[tasa].sum().idxmax()
    
    #Devuelvo la mayor tasa.
    tasa = por_delito[tasa].max()
    
    #Retorna las dos variables en forma de tupla.
    return año, tasa

"""
Prueba de las funciones anteriores.

tasa_hechos = menor_tasa("tasa_hechos", "Homicidios dolosos", df)
tasa_victimas = menor_tasa("tasa_victimas", "Homicidios dolosos", df)

"""

#%% 4_ ¿En qué año se registraron más crímenes en total? ¿Y menos?

#Devuelve un dataframe 2x2 contiene año con mayor y menor cantidad de hechos delictivos registrados. 
def mas_y_menos_crimenes_total(df): 

    #Agrupa por año y suma los valores de la columna "cantidad_de_hechos" y selecciona el año de registro máximo.
    año_mas = df.groupby(["anio"])["cantidad_hechos"].sum().idxmax()
    
    #Agrupa por año y suma los valores de la columna "cantidad_de_hechos" y selecciona el año de registro mínimo.
    año_menos = df.groupby(["anio"])["cantidad_hechos"].sum().idxmin()
    
    #Agrupa por año y suma los valores de la columna "cantidad_de_hechos" y selecciona el registro máximo.
    cantidad_mayor = df.groupby(["anio"])["cantidad_hechos"].sum().max() 
    
    #Agrupa por año y suma los valores de la columna "cantidad_de_hechos" y selecciona el registro mínimo.
    cantidad_menor = df.groupby(["anio"])["cantidad_hechos"].sum().min() 
    
    #Utilizo el metodo dict. para eso, creo el diccionario, cuyos valores son las variables asignadas anteriormente.
    datos_diccionario = {"año":[año_mas, año_menos], "cantidad_hechos":[cantidad_mayor,cantidad_menor]}

    #Creó un Data Frame con las claves del dict como columnas y un index literal.
    #De esta forma, cada fila muestra el año correspondiente y la cantidad de hechos cometidos.
    resultado_df = pd.DataFrame(datos_diccionario, index=["Mayor cantidad de hechos", "Menor cantidad de hechos"])
    
    #Retorna el dataframe
    return resultado_df

"""
Prueba de las función anterior.

mas_menos_crimenes = mas_y_menos_crimenes_total(df)

"""

#%% 5_ ¿En qué tipo de delitos la tasa de víctimas femeninas supera a la tasa masculina?

#Devuelve en qué tipo de delitos la tasa de víctimas femeninas supera a la tasa masculina en un año determinado.
def relacion_tasa_vict_fem_mas(año, df): 
    
    #Defino en una variable, un dataframe con los datos únicamente del año escogido, usando un filtro.
    año_datos=df[df["anio"]==año]
    
    #Comparó las tasas de víctimas femenina y masculina, quedandome con los nombres de los delitos en los cuales la 1ra
    #es mayor que la 2da. Además del nombre del delito, recuperar las tasas en sí, tanto para chequear como para comparar.
    tipos_delito = año_datos[año_datos["tasa_victimas_fem"] > año_datos["tasa_victimas_masc"]][["codigo_delito_snic_nombre", "tasa_victimas_fem", "tasa_victimas_masc"]]
    return tipos_delito

"""
Prueba de las función anterior.

femmasc = relacion_tasa_vict_fem_mas(2019, df)

"""

#%% 6_ ¿Cuál es el delito con la mayor cantidad de víctimas sin definir su género (sd)?

#Devuelve el crimen con mayor cantidad de víctimas, del tipo introducido, y la cantidad.
def mayor_victimas(victima, df):
    
    #Agrupa por tipo de crimen y suma las cantidad de víctimas del tipo solicitado y devolviendo el nombre del máximo.
    el_mayor = df.groupby(["codigo_delito_snic_nombre"])[victima].sum().idxmax()
    
    #Una vez conseguido el nombre del crimen agrupó un df nuevo con ese crimen.
    el_delito = df[df.codigo_delito_snic_nombre == el_mayor]
    
    #Y de ese df retorno el año con más víctimas.
    maximo = el_delito[victima].max()
    
    #El número de dicho año
    año = el_delito[el_delito[victima] == maximo].anio
    
    #Y la suma total de las victimas hasta ahora de ese delito.
    cant_total = el_delito[victima].sum()
    
    #Retorna las var anteriores en forma de tupla.
    return el_mayor, maximo, año, cant_total

"""
Prueba de las función anterior.

mayor_victimas_sd = mayor_victimas("cantidad_victimas_sd", df)

"""

#%% 7_ Generar un dataframe donde, para cada delito, se obtenga el promedio de la cantidad de hechos a lo largo de los años, con su desvío estándar correspondiente.

#Devuelve un nuevo df con dos columnas, una para promedio y otra para el desvío de la columna solicitada.
def promedio_sdt_columna(columna,df):
    
    #Creó la serie con el promedio de la columna solicitada.
    promedio = df.groupby(["codigo_delito_snic_nombre"])[columna].mean()
    
    #Creó la serie con el desvío de la columna solicitada.
    desvio  = df.groupby(["codigo_delito_snic_nombre"])[columna].std()
    
    #Las concaten.
    df_prom_sdt = pd.concat([promedio,desvio], axis=1,keys=["promedio", "desvio"])
    
    #Retorno el df.
    return df_prom_sdt

"""
Prueba de las función anterior.
    
promedio_sdt_hechos = promedio_sdt_columna("cantidad_hechos", df)

"""

#%% 8_ Agregar una columna con el año de más prevalencia de cada delito, con la cantidad de veces que sucedió y cantidad de víctimas.

#Devuelve un dataframe con 4 columnas("codigo_delito_snic_nombre", "anio", "cantidad_victimas", "cantidad_hechos")
def agregar(df):
    
    #Preparamos una reducción del dataset con los datos que nos interesan.
    datos = df.loc[:,["codigo_delito_snic_nombre","anio","cantidad_victimas","cantidad_hechos"]]
    
    #Creamos una serie con los indices de los maximos de cada crimen.
    indice = datos.groupby("codigo_delito_snic_nombre")["cantidad_hechos"].idxmax()
    
    #Preparamos el nuevo dataset con los indices que queremos guardar.
    df_final = pd.DataFrame({
        "codigo_delito_snic_nombre": ["str"],
        "anio":[0],
        "cantidad_victimas":[0],
        "cantidad_hechos":[0],
        })

    #Iteramos la serie de los índices
    for n in range(len(indice)):
        
        #Igualamos "a" al indice
        a = indice[n] 
        
        #Igualamos esta variable a un fila del del dataset.
        #De esta forma nos queda un elemento tipo dataset y no serie, que nos facilita la creación del nuevo dataframe.
        dato_crimen = datos.loc[[a]]
        
        #Concatenamos cada fila en el nuevo dataset.
        df_final = pd.concat([df_final, dato_crimen], axis=0, ignore_index=True) 
    
    #Eliminamos la primera fila que contiene los datos que usamos solo para crear el df_final.
    df_final = df_final.drop([0])
    
    #Reseteamos el índice del dataset
    df_final = df_final.reset_index()
    
    #Eliminamos fila extra.
    df_final = df_final.drop("index", axis = 1)
    
    #igualamos las etiquetas nuevas
    df_final.columns = ["delito", "anio_mas_prevalencia", "cantidad_victimas_mas_prevalencia", "cantidad_hechos_mas_prevalencia"]
    
    #Concatenamos en el eje horizontal
    df_final = pd.concat([df,df_final], axis=1, sort = False)
    
    #Retornamos el df nuevo igual al original pero con estas columnas agregadas
    return df_final

"""
Prueba de las función anterior.
    
dffinal = agregar(df)

"""

#%% Consignas Clase II

"""
    Nombre: Delitos CABA

    Descripción: Información de delitos para todo el país. Acá sólo usaremos los datos de la Ciudad de Buenos Aires para 2022

    Preguntas:

        Clase II

            Agregar dos columnas al DataFrame original: la proporción de víctimas masculinas/víctimas totales y víctimas femeninas/victimas totales.
            Averiguar si existe una correlación entre la tasa de víctimas masculinas y femeninas. Graficar y obtener el coeficiente de Pearson (r).
            Hacer una comparación de medias de cantidad de víctimas por delito entre homicidios dolosos y abusos sexuales.
"""

#%% 1_Agregar dos columnas al Data Frame original: la proporción de víctimas masculinas/víctimas totales y víctimas femeninas/victimas totales.

#Devuelve el resultado de dividir dos cantidades presentes a dos columnas del df. Auxiliar de la función siguiente.
def prop_cantidades(cantidad_1, cantidad_2, df):
    
    return df[cantidad_1] / df[cantidad_2]

#Utilizando la función de prop_cantidades, agrega dos columnas con el resultado de las víctimas femeninas y masculinas entre las víctimas totales.
#Esta columna presenta valores reales solo donde los datos existen y las víctimas totales no son cero.
def agregar_prop(df): 
    
    #Hacemos una copia del df, para no modificar los datos originales.
    df_con_prop = df.copy()

    #Calculamos la proporción de víctimas masc/fem sobre el total, utilizando la función anterior. 
    #Estas dos funciones operan en las columnas, a nivel serie, no fila por fila.
    df_con_prop["prop_vict_masc"] = prop_cantidades("cantidad_victimas_masc", "cantidad_victimas", df_con_prop)
    df_con_prop["prop_vict_fem"] = prop_cantidades("cantidad_victimas_fem", "cantidad_victimas", df_con_prop)

    #Reemplazamos inf y -inf con NaN para evitar resultados no definidos. Esta linea fue agregada apra evitar los problemas en division por cero.
    df_con_prop.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    #Retornamos el nuevo df con las columnas agregadas
    return df_con_prop

"""
Prueba de las función anterior.

df_nuevo = agregar_prop(df)

"""

#%% 2_Averiguar si existe una correlación entre la tasa de víctimas masculinas y femeninas. Graficar y obtener el coeficiente de Pearson (r).


#Toma dos variables, hace un scatter calcula la regresión lineal y la gráfica
#también devuelve el coeficiente de pearson(p) y evalúa si existe o no correlación
#dado un criterio que determinamos.
def correlacion(var1, var2, df):

    hay = False
    
    #Eliminamos NaNs de los valores de ambas variables.
    df_sin_nan = df.dropna(subset=[var1, var2])

    #Crea el gráfico de puntos discretos (scatter()).
    ax = df_sin_nan.plot.scatter(x=var1, y=var2, label='Datos', color='blue', alpha=0.5)
    
    #Calculamos todas las variables de la regresión lineal usando scipy.
    pend, orig, coeficiente_pearson, p, err = st.linregress(df_sin_nan[var1], df_sin_nan[var2])
    
    #Graficamos la línea de regresión en el gráfico scatter.
    ax.plot(df_sin_nan[var1], orig + pend * df_sin_nan[var1], color="orange", label='Regresión Lineal')
    
    #Añadimos título y etiquetas de los ejes.
    ax.set_title(f'Relación entre {var1.replace("_"," ")} y {var2.replace("_"," ")}', fontweight='bold')
    ax.set_xlabel(f'{var1.replace("_"," ")} (Unidades)')
    ax.set_ylabel(f'{var2.replace("_"," ")} (Unidades)')
    
    #Añade la cuadrícula Y las leyendas.
    ax.grid(True)
    ax.legend()
    
    #Ponemos el valor de p y el coeficiente de Pearson (r) fuera del gráfico, abajo izq.
    plt.figtext(0, -0.05, f'Coeficiente de Pearson (r): {coeficiente_pearson:.2f}\np-valor: {p:.2e}', fontsize=10)
    
    #Ajusta la distribución para que no se solapen los elementos de la gráfica.
    plt.tight_layout()
    
    #Mostrar la gráfica.
    plt.show()
    
    #Determinar si hay una correlación significativa usando el criterio 
    #r> 0.7, para afirmar si existe una correlación fuerte
    if coeficiente_pearson > 0.7:
        hay = True
    
    return hay, coeficiente_pearson

"""
Prueba de las función anterior.
    
correlacion("cantidad_victimas_masc", "cantidad_victimas_fem",df)

"""

#%% 3_Hacer una comparación de medias de cantidad de víctimas por delito entre homicidios dolosos y abusos sexuales.

#Devuelve gráfico con dos boxplots con la distribución de la cantidad 
#de víctimas para dos delitos cuyas medidas y cuantiles se desean comparar.
def boxplot_medias_delitos(df, columna, delito1, delito2):
    
    #Filtramos el df para eliminar NaNs, ceros y filtramos por los delitos a comparar.
    df_sin_nan = df.dropna(subset=[columna])
    df_filtrado = df_sin_nan[df_sin_nan[columna] != 0]
   
    #Dividimos la figura en dos partes (1 fila, 2 columnas, se numeran desde el 1)
    #figsize ajusta el tamaño de la figura.
    plt.figure(figsize=(10, 5))  
    
    #CÓDIGO PARA EL BOXPLOT DELITO 1:
    #localiza el boxplot. hay una fila, dos columnas y estamos poniendo el 
    #primer boxplot en la primera columna.
    plt.subplot(1, 2, 1)
    
    #Filtró el df por delito 1 y llamo a la función boxplot básica.
    df_delito1 = df_filtrado[df_filtrado["codigo_delito_snic_nombre"] == delito1]
    df_delito1.boxplot(column=[columna])
    
    #Utilizó Normal Distribution para descomprimir los datos en el eje x. 
    #Moviéndose de forma aleatoria hacia los laterales.
    xs_delito1 = np.random.normal(1, 0.05, df_delito1.shape[0])
    
    #Tomo todos datos de la columna y los convierto en una lista para definir el eje y.
    ys_delito1 = df_delito1[columna].to_list()
    
    #Añado los parámetros xs e ys, ploteando. Alpha 0.7 es el color de 
    #los puntos individuales.
    plt.plot(xs_delito1, ys_delito1, ".", alpha=0.7)
    
    #Defino el promedio para añadirlo como etiqueta y como número explicito.
    media_delito1 = df_delito1[columna].mean()
    plt.plot([1], [media_delito1], "o", label=f'Media: {media_delito1:.2f}', color='blue')
    
    #Añade el numero alineado con el punto de media.
    plt.text(1.2, media_delito1, f'{media_delito1:.2f}', va='center', ha='left', color='blue')
    
    #Definir el valor de la mediana .median() y la añado como etiqueta con su valor.
    mediana_delito1= df_delito1[columna].median()
    plt.plot([1], [mediana_delito1], "-", label=f'Mediana: {mediana_delito1:.2f}', color='green')  #Media para delito1.
    
    #Defino 3er cuartil (75% de los datos) y lo añado como etiqueta con su valor.
    cuartil_3_del1= df_delito1[columna].quantile(0.75)
    plt.plot([1], [cuartil_3_del1], "v", label=f'3er cuartil:{cuartil_3_del1:.2f}', color="Orange")
    
    #Títulos
    plt.title(f'{delito1}', size=10, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Víctimas registradas (unidades)')
    plt.legend()
    
    #CÓDIGO BOXPLOT 2:
    #Localiza el boxplot en la figura (segunda columna fila 1).
    plt.subplot(1, 2, 2)
    
    #Filtró el df por delito 2 y llamó a la función boxplot básica.
    df_delito2 = df_filtrado[df_filtrado["codigo_delito_snic_nombre"] == delito2]
    df_delito2.boxplot(column=[columna], showfliers=True)
    
    #Definición de las variables xs e ys para el delito 2. Mismo método.
    xs_delito2 = np.random.normal(1, 0.05, df_delito2.shape[0])
    ys_delito2 = df_delito2[columna].to_list()
    plt.plot(xs_delito2, ys_delito2, ".", alpha=0.7)
    
    #Definición de media: cálculo, punto, etiqueta y valor alineado al punto.
    media_delito2 = df_delito2[columna].mean()
    plt.plot([1], [media_delito2], "o", label=f'Media: {media_delito2:.2f}', color='red')  #Media para delito2.
    plt.text(1.2, media_delito2, f'{media_delito2:.2f}', va='center', ha='left', color='red')
    
    #Definir el valor de la mediana .median() y la añado como etiqueta con su valor.
    mediana_delito2= df_delito2[columna].median()
    plt.plot([1], [mediana_delito2], "-", label=f'Mediana: {mediana_delito2:.2f}', color='green')
    
    #Defino 3er cuartil (75% de los datos), añado su etiqueta y valor.
    cuartil_3_del2= df_delito2[columna].quantile(0.75)
    plt.plot([1], [cuartil_3_del2], "v", label=f'3er Cuartil: {cuartil_3_del2:.2f}', color="Orange")
    
    #Títulos.
    plt.title(f'{delito2}', size=10, fontweight='semibold')
    plt.xlabel('')
    plt.ylabel('Víctimas registradas (unidades)')
    plt.legend()
    
    #GRAFICACIÓN FINAL 
    #Añadimos el título final encima de los boxplots, reguló el tamaño de fuente 
    plt.suptitle('Comparacion de distribución de cantidad de víctimas registradas anualmente', fontsize=12, fontweight='bold')
    
    #Ajustamos la posición de los subplots en el con plt.tight_layout() 
    #y mostramos el gráfico final.
    plt.tight_layout()
    plt.show()

"""
Prueba de las funciones anteriores.

hacer_boxplot_comp_medias_delito(df, 'cantidad_victimas', "Abusos sexuales con acceso carnal (violaciones)","Homicidios dolosos")

"""

#%% COMPLEMENTARIOS

#%% Crimenes mas y menos cometidos

#Agrupamos los datos por tipo de delito y sumamos la cantidad de hechos en total entre 2000 y 2022
delitos = df.groupby("codigo_delito_snic_nombre")["cantidad_hechos"].sum().reset_index()
delitos = delitos.sort_values(by="cantidad_hechos", ascending=True)

#Creamos las variables de los 10 delitos más cometidos y los 10 menos cometidos
delitos_10 = delitos.tail(10)
delitos_10_menos = delitos.head(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(delitos_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes mas cometidos", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por millon")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Contra Libertad", "Leyes Especiales", "Contra Admin Pub", "Tentativa Robo",
             "Estupefacientes", "Accidentes Viales", "Contra propiedad",
             "Amenazas" "Lesiones", "Hurto" ]

#Graficamos.
ax.bar(coordenadas, delitos_10["cantidad_hechos"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#Creamos la segunda figura con los diez crímenes menos cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes menos cometidos", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por unidad")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Venta Armas", "contrabando Nuclear", "Desvio Drogas", "Delitos Migratorios",
             "Marcaje productos", "Ley Fauna", "Confabulacion Drogas",
             "Fabricacion Armas", "Trata Personas", "Financiacion Drogas" ]

#Graficamos.
ax.bar(coordenadas,delitos_10_menos["cantidad_hechos"], ancho, tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#%% Crimenes en promedio más cometidos

promedio_sdt_hechos = promedio_sdt_columna("cantidad_hechos", df)
delitos = promedio_sdt_hechos.sort_values(by="promedio", ascending=True)

#Creamos las variables de los 10 delitos más cometidos y los 10 menos cometidos
delitos_10 = delitos.tail(10)
delitos_10_menos = delitos.head(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(delitos_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes mas cometidos en promedio", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Promedio de Hechos")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Leyes n,c,p", "Tentativa robo", "Tenencia droga", "Estupefacientes",
             "Accidentes viales", "Contra la propiedad", "Amenazas", "Lesiones", "Hurto", "Robo" ]

#Graficamos.
ax.bar(coordenadas, delitos_10["promedio"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#Creamos la segunda figura con los diez crímenes menos cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes menos cometidos en promedio", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Promedio de Hechos")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Venta Armas", "contrabando Nuclear", "Desvio Drogas", "Delitos Migratorios",
             "Ley Fauna", "Marcaje productos", "Contra la seguridad\nnacional",
             "Confabulacion de drogas", "Contra el estado\ncivil", "Fabricación de armas" ]

#Graficamos.
ax.bar(coordenadas,delitos_10_menos["promedio"], ancho, tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#%% Evolución de la cantidad de hechos y la cantidad de víctimas

#Creamos los dataframe con la cantidad de hechos y victimas totales agrupados por años.
cantidad_hechos = df.groupby("anio")["cantidad_hechos"].sum().reset_index()
cantidad_victimas = df.groupby("anio")["cantidad_victimas"].sum().reset_index()

#Creamos las etiquetas para el eje x.
etiquetas = cantidad_hechos["anio"]

#Gráfico por separado de la evolución individual de la cantidad de hechos y la cantidad de víctimas.

#Creamos las figuras donde irán los gráficos con sus dimensiones.
fig, ax = plt.subplots()
fig.set_size_inches(15,5)

#Personalizamos el título, la escala, las etiquetas y la grilla.
fig.suptitle("Evolucion de la Cantidad de Hechos", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por unidad")
plt.xlabel("Años")
ax.grid()

#Realizamos el gráfico.
ax.plot(cantidad_hechos["cantidad_hechos"], marker = "o", label = "cantidad_hechos")
ax.set_xticks(range(len(cantidad_hechos)), etiquetas, rotation = 70)

#Ejecutamos la figura.
plt.show()


#Creamos las figuras donde iran los graficos con sus dimenciones.
fig, ax = plt.subplots()
fig.set_size_inches(15,5)

#Personalizamos el título, la escala, las etiquetas y la grilla.
fig.suptitle("Evolucion de la Cantidad de victimas", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por unidad")
plt.xlabel("Años")
ax.grid()

#Realizamos el grafico.
ax.plot(cantidad_victimas["cantidad_victimas"], marker = "v", label = "cantidad_victimas")
ax.set_xticks(range(len(cantidad_hechos)), etiquetas, rotation = 70)

#Ejecutamos la figura.
plt.show()

#Creamos las figuras donde irán los gráficos con sus dimensiones.
fig, ax = plt.subplots()
fig.set_size_inches(15,5)

#Personalizamos el título, la escala, las etiquetas y la grilla.
fig.suptitle("Evolucion de la Cantidad de hechos y victimas", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos y victimas por unidad")
plt.xlabel("Años")
ax.grid()

#Realizamos el gráfico, en este caso con las dos variables.
ax.plot(cantidad_hechos["cantidad_hechos"], marker = "o", label = "cantidad_hechos")
ax.set_xticks(range(len(cantidad_hechos)), etiquetas, rotation = 70)
ax.plot(cantidad_victimas["cantidad_victimas"], marker = "v", label = "cantidad_victimas")
ax.set_xticks(range(len(cantidad_hechos)), etiquetas, rotation = 70)

#Ejecutamos la figura con las leyendas de las dos líneas.
plt.legend()
plt.show()

#%% Grafico de la diferencia entre las víctimas fem y masc

def cantidad_fem_masc(df): 
    #Comparó las tasas de víctimas femenina y masculina, quedandome con los nombres de los delitos en los cuales
    #La 1ra es mayor que la 2da. Además del nombre del delito, me interesa recuperar las tasas en sí, tanto para chequear como para comparar.
    tipos_delito = df[df["cantidad_victimas_fem"] > df["cantidad_victimas_masc"]][["codigo_delito_snic_nombre", "cantidad_victimas_fem", "cantidad_victimas_masc"]]
    return tipos_delito

masc = df.groupby("codigo_delito_snic_nombre")["cantidad_victimas_masc"].sum().reset_index()

fem = df.groupby("codigo_delito_snic_nombre")["cantidad_victimas_fem"].sum().reset_index()

cantidad_completas =  pd.concat([fem, masc], axis=1, ignore_index=True)
cantidad_completas = cantidad_completas.drop(2,axis=1)
cantidad_completas = cantidad_completas.sort_values(by=1, ascending=True)
cantidad_completas.columns = ["codigo_delito_snic_nombre", "cantidad_victimas_fem", "cantidad_victimas_masc"]

datos = cantidad_fem_masc(cantidad_completas)
ancho = 0.4
coordenadas = np.arange(len(datos))
etiquetas = ["Trata de personas\nagravado", "Trata de personas\nsimple", "Abusos sexuales", "Delitos contra\nla integridad sexsual"]

fig, ax = plt.subplots()
fig.set_size_inches(7,7)

fig.suptitle("Comparacion de Cantidad de Victimas", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de victimas por unidad")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 45)

fem_graf = ax.bar(coordenadas + ancho/2 , datos["cantidad_victimas_fem"], ancho, tick_label = etiquetas, label = "cantidad_victimas_fem") 
masc_graf = ax.bar(coordenadas - ancho/2 , datos["cantidad_victimas_masc"], ancho, tick_label = etiquetas, label = "cantidad_victimas_masc") 

ax.bar_label(fem_graf, padding=3)
ax.bar_label(masc_graf, padding=3)

fig.tight_layout()
plt.legend()
plt.show()

#%% Diferencia pre y post pandemia

pre_pandemia = df[df.anio <= 2019]
post_pandemia = df[df.anio > 2020]
pandemia= df[df.anio == 2020]

pre_pandemia = pre_pandemia.groupby("codigo_delito_snic_nombre")["cantidad_hechos"].sum().reset_index()
pre_pandemia = pre_pandemia.sort_values(by="cantidad_hechos", ascending=True)

pandemia = pandemia.groupby("codigo_delito_snic_nombre")["cantidad_hechos"].sum().reset_index()
pandemia = pandemia.sort_values(by='cantidad_hechos', ascending=True)

post_pandemia = post_pandemia.groupby("codigo_delito_snic_nombre")["cantidad_hechos"].sum().reset_index()
post_pandemia = post_pandemia.sort_values(by="cantidad_hechos", ascending=True)

#%% Pre pandemia

#Creamos las variables de los 10 delitos más cometidos y los 10 menos cometidos
pre_10 = pre_pandemia.tail(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(pre_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes mas cometidos Pre-Pandemia", fontsize = 15)
plt.title("Entre 2000 - 2019", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por millon")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Contra Admin Pub", "Leyes Especiales", "tentativa Robo", "Contra la Propiedad",
             "Estupefacientes", "Accidentes Viales", "Amenazas", "Lesiones", "Hurto", "Robo" ]

#Graficamos.
ax.bar(coordenadas, pre_10["cantidad_hechos"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()



#%% en pandemia

en_10 = pandemia.tail(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(pre_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes mas cometidos en Pandemia", fontsize = 15)
plt.title("2020", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por unidad")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Accidentes Viales", "Tenencia de Drogas", "Amenazas", "Contra la Propiedad",
             "Lesiones", "Contra la Seguridad Pub\n(otros delitos)", "Contra la Seguridad Pub",
             "Estupefacientes", "Hurto", "Robo" ]

#Graficamos.
ax.bar(coordenadas, en_10["cantidad_hechos"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#%% post pandemia

#Creamos las variables de los 10 delitos más cometidos y los 10 menos cometidos
post_10 = post_pandemia.tail(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(pre_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes mas cometidos Post-Pandemia", fontsize = 15)
plt.title("Entre 2020 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Hechos por unidad")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Contra Admin Pub", "Tentativa Robo", "Tenencia de Drogas", "Accidentes Viales",
             "Amenazas", "Estupefacientes", "Lesiones", "Contra la Propiedad", "Hurto", "Robo" ]

#Graficamos.
ax.bar(coordenadas, post_10["cantidad_hechos"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#%% Mas victimas

#Agrupamos los datos por tipo de delito y sumamos la cantidad de hechos en total entre 2000 y 2022
delitos = df.groupby("codigo_delito_snic_nombre")["cantidad_victimas"].sum().reset_index()
delitos = delitos.sort_values(by="cantidad_victimas", ascending=True)

#Creamos las variables de los 10 delitos más cometidos y los 10 menos cometidos
delitos_10 = delitos.tail(10)

#Establecemos las coordenadas y el ancho de las barras a graficar
coordenadas = np.arange(len(delitos_10))
ancho =  0.8

#Creamos la primera figura con los diez crímenes más cometidos.
fig, ax = plt.subplots()
fig.set_size_inches(10,4)

#Personalizamos el título, la escala y las etiquetas.
fig.suptitle("Los 10 Crimenes con mas Victimas", fontsize = 15)
plt.title("Entre 2000 - 2022", style = "italic", fontsize = 9)
plt.ylabel("Cantidad de Victimas")
plt.xlabel("Tipos de delitos")
plt.xticks(rotation = 70)
etiquetas = ["Homicidio doloso", "Suicidios", "Abuso sexual", "Contra la Libertad\n(n,c,p)",
             "Contra la inegridad\nSexual", "Contra la Persona", "Lesiones", "Contra la Libertad",
             "Accidentes Viales", "Lesiones Dolosas" ]

#Graficamos.
ax.bar(coordenadas, delitos_10["cantidad_victimas"], ancho , tick_label = etiquetas)

#Ejecutamos la figura.
plt.show()

#%% Distribución fem y masc

def hacer_boxplot_comp_medias(df, columna, delito):
    
    #Filtramos el DataFrame para eliminar NaNs, ceros y filtramos por el delito especificado.
    df_sin_nan = df.dropna(subset=[columna])
    df_filtrado = df_sin_nan[df_sin_nan[columna] != 0]
    df_filtrado = df_filtrado[df_filtrado["codigo_delito_snic_nombre"] == delito]
    
    #Ajustamos el tamaño de la figura antes de crear el boxplot.
    plt.figure(figsize=(6, 4))
    df_filtrado.boxplot(column=[columna], showfliers=False)
    
    #Añadimos puntos individuales y medias.
    xs = np.random.normal(1, 0.05, df_filtrado.shape[0])
    ys = df_filtrado[columna].to_list()
    
    plt.plot(xs, ys, ".", alpha=0.5, color='blue')  #Puntos individuales
    
    media = df_filtrado[columna].mean()
    plt.plot([1], [media], "o", label=f'Media: {media:.2f}', color='orange')  #Media
    
    mediana = df_filtrado[columna].median()
    plt.plot([1], [mediana], "-", label=f'Mediana: {mediana:.2f}', color='green')  #Mediana
    
    cuartil_3= df_filtrado[columna].quantile(0.75)
    plt.plot([1], [cuartil_3], "v", label=f'Mediana: {cuartil_3:.2f}', color="violet")
    
    #Añadimos títulos y etiquetas
    plt.suptitle(f'Distribución de {columna.replace("_", " ")}', fontweight='bold')
    plt.title(f' para {delito}', fontsize=10)
    plt.xlabel('')
    plt.ylabel('Víctimas Registradas (Unidades)')
    plt.legend()
    
    #Ajustamos y mostramos el boxplot completo
    plt.tight_layout()
    plt.show()
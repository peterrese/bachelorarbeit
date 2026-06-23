import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=120) 
#  Anpassbar Nachkommastellen drucken, um Debugging zu ermöglichen.
#120 Zeichen pro Zeile 




def build_extended_markov_chain(n1: int, n2: int, m: int) -> np.ndarray:
    """
    Baut die gesamte Übergangsmatrix mit den Zuständen = (matrix, current_state).
    """
    #assert m % 2 == 0, "Interim matrix size m must be even"
    #assert m >= 4, "Interim matrix size m must be at least 4"
    M1 = create_markov_matrix(n1)   # Form (n1+1, n1+1)
    M2 = create_markov_matrix(n2)   # Form (n2+1, n2+1)
    print(M2)
    exit()
    Interim = np.full((m, m), 1.0 / m)   # Gleichverteilung der innerne Matrix

    
    # Hier legen wir jetzt fest, wo die einzelnen Blöcke in der großen Gesamt-Tabelle/Matrix platziert werden sollen (oberer Block, mittlerer Block, unterer Block).
    off1 = 0
    #Der erste Block (Tabelle 1) fängt bei der Zeilen- und Spaltennummer 0 an.
    off_interim = n1 + 1
    #Der mittlere Block fängt genau hinter dem ersten Block an.
    off2 = off_interim + m
    #Der untere Block fängt genau hinter dem mittleren Block an. Dazu nehmen wir die Startposition des mittleren Blockes+ deren Größe.

    
    # Definiert die Gesamtgröße der Matrix
    total_states = (n1 + 1) + m + (n2 + 1)
    P = np.zeros((total_states, total_states))
    #Erstellt eine 2-D Liste die so groß ist wie unsere Matrix und setzt die Einträge = 0

    # ----- Matrix 1(A) übergänge -----
    for curr in range(n1 + 1):
        #Zwischen den Einträgen 0 bis n1 
        src = off1 + curr #Verschieben der Reihe
        #Überprüfen der genauen Zeilennummer der großen Tabelle P
        if curr == n1:   # Überprüfen ob es der letzte Zustand ist(n1)
            p_self = M1[curr, curr] #ÜbergangsWahrscheinlichkeit in sich selbst wird als p_self gemerkt
            if p_self > 0: #Wenn sie größer als 0 ist 
                P[src, off_interim + 0] += p_self #Wird in den ersten Zustand der kleinen Übergangsmatrix (A_q aufbau verbunden mit B/E) in die große P übertragen
            for nxt in range(n1 + 1): #Für alle anderen Zustände in der ersten Matrix, immer noch für den Übergangszustand zu Interim
                if nxt == curr: #haben wir oben gesetzt
                    continue #Überspringen wenn wir wieder im Zustand sind wie davor
                prob = M1[curr, nxt] # Die Wahrscheinlichkeit auß der kleinen Matrix 1
                if prob > 0: # Wenn eine Wahrscheinlichkeit existiert >0
                    P[src, off1 + nxt] += prob # Wird diese Übernommen in die große Matrix P
        else: #Wenn der aktuelle Zustand nicht der letzte Zustand der kleinen Matrix 1 ist
            for nxt in range(n1 + 1):
                prob = M1[curr, nxt] 
                if prob > 0: #überprüft die Wahrscheinlichkeiten ob sie größer als 0 ist
                    P[src, off1 + nxt] += prob #Trägt diese in die große Tabelle/Matrix P ein

    # ----- Matrix 2(C) Übergänge -----
    #Gleicher Code bloß mit der Indize verschiebung (off2)
    for curr in range(n2 + 1):
        src = off2 + curr
        if curr == n2:   # Äußerer Punkt der zweiten Randmatrix, letzte Iteration der Schleife
            p_self = M2[curr, curr]
            if p_self > 0:
                # Übergangspunkt wird in den letzen Punkt m-1 geschickt
                P[src, off_interim + (m - 1)] += p_self
            for nxt in range(n2 + 1):
                if nxt == curr:
                    continue
                prob = M2[curr, nxt]
                if prob > 0:
                    P[src, off2 + nxt] += prob
        else:
            for nxt in range(n2 + 1):
                prob = M2[curr, nxt]
                if prob > 0:
                    P[src, off2 + nxt] += prob

    # ----- Interim/übergangsmatrix(B oder E) Verteilungen -----
    for curr in range(m):
        # Verschiebt in Gesamtmatrix für aktuellen Interim-Zustand
        src = off_interim + curr
        if curr == 0: #für den ersten Eintag der Übergangsmatrix
            p_self = Interim[curr, curr] #Die Wahrscheinlichkeit in sich zurückzukehren
            P[src, off1 + n1] += p_self # Schickt die Wahrscheinlichkeit in den äußersten Rand von Matrix 1/A
            for nxt in range(m):
                if nxt == curr:
                    continue
                prob = Interim[curr, nxt]
                P[src, off_interim + nxt] += prob
        elif curr == m - 1: #Gleiche vorgehen wie mit dem Übergangspunkt in Matrix 1/A nur nun in Matrix 2/B
            p_self = Interim[curr, curr]
            P[src, off2 + n2] += p_self
            for nxt in range(m):
                if nxt == curr:
                    continue
                prob = Interim[curr, nxt]
                P[src, off_interim + nxt] += prob
        else: # Übertragen der Wahrscheinlichkeiten der anderen Matrix eintäge normal
            for nxt in range(m):
                prob = Interim[curr, nxt]
                if prob > 0:
                    P[src, off_interim + nxt] += prob

    # überprüft, ob alle Zeilen sich zu 1 addieren
    row_sums = P.sum(axis=1)
    # Wenn nicht gibt der Code die Zeile an, an der die Summe ungleich 1 ist
    assert np.allclose(row_sums, 1.0), f"Zeile summiert sich nicht zu 1: {row_sums}"
    #Gibt die Übergangsmatrix P wieder.
    return P

def build_extended_markov_chain_v2(n1: int, n2: int, m: int) -> np.ndarray:
    """
    Erweiterte Markow-Kette mit drei Blöcken:
    - Matrix 1/A (Größe n1+1)
    - Interim B/E (Größe m, erzeugt durch build_uniform_interim_matrix)
    - Matrix 2/B (Größe n2+1)
    """
    #assert m % 2 == 0, "Interim matrix size m must be even"
    #assert m >= 4, "Interim matrix size m must be at least 4"

    #obere Markov-Kette(links oben)
    M1 = create_markov_matrix(n1)   # wird extern definiert
    #untere Markov-Kette(rechts unten)
    M2 = create_markov_matrix(n2)


    Interim = build_weighted_interim_matrix(m)  # Baut die Übergangsmatrix E (Gewichtet)
    
    
    # Offsets für die Blöcke in der Gesamtmatrix(A, E, B) wie bei dem Bau der Übergänge
    off1 = 0
    off_interim = n1 + 1
    off2 = off_interim + m

    #Definieren der Gesamtgröße der Matrix
    total_states = (n1 + 1) + m + (n2 + 1)
    #Gesamtmatrix wird übergeben
    P = np.zeros((total_states, total_states))

    # ----- Block 1 (Matrix 1) -----
    #Von Start der oberen Matrize bis n1+1
    for curr in range(n1 + 1):
        #derzeitigen index in Gesamtmatrix berechnen
        src = off1 + curr
        #curr ist der aktuelle Zustand in Matrix 1, src ist der entsprechende Index in der Gesamtmatrix
        if curr == n1:   # äußerer Zustand
            p_self = M1[curr, curr]
            if p_self > 0:
                P[src, off_interim + 0] += p_self # Übernimmt Wahrscheinlichkeit für Selbstschleife im äußeren Zustand von Matrix 1. 
            # Übergänge zu anderen Zuständen in Matrix 1
            for nxt in range(n1 + 1):
                if nxt == curr:
                    continue
                prob = M1[curr, nxt]
                if prob > 0:
                    P[src, off1 + nxt] += prob
        else: # Wenn nicht äußerer Zustand, einfach Übergänge zu anderen Zuständen in Matrix 1 übernehmen 
            for nxt in range(n1 + 1):
                prob = M1[curr, nxt]
                if prob > 0:
                    P[src, off1 + nxt] += prob #Übernahme aus kleiner M1 in die große P, mit entsprechendem Offset für Matrix 1

    # ----- Block 2 (Matrix 2) -----
    for curr in range(n2 + 1):
        src = off2 + curr
        if curr == n2:   # äußerer Zustand
            p_self = M2[curr, curr]
            if p_self > 0:
                P[src, off_interim + (m - 1)] += p_self
            for nxt in range(n2 + 1):
                if nxt == curr:
                    continue
                prob = M2[curr, nxt]
                if prob > 0:
                    P[src, off2 + nxt] += prob
        else:
            for nxt in range(n2 + 1):
                prob = M2[curr, nxt]
                if prob > 0:
                    P[src, off2 + nxt] += prob

    # ----- Interim Block (übergangsmatrix E) -----
    for curr in range(m):
        src = off_interim + curr
        if curr == 0:
            # Selbstschleife (m-1)/m führt zu Matrix 1, äußerer Zustand
            p_self = Interim[curr, curr]   # = (m-1)/m
            if p_self > 0:
                P[src, off1 + n1] += p_self
            # Übergang zu Interim-Zustand 1 (falls vorhanden)
            p_next = Interim[curr, 1]      # = 1/m
            if p_next > 0:
                P[src, off_interim + 1] += p_next
        elif curr == m - 1:
            p_self = Interim[curr, curr]   # = (m-1)/m
            if p_self > 0:
                P[src, off2 + n2] += p_self
            p_prev = Interim[curr, m-2]    # = 1/m
            if p_prev > 0:
                P[src, off_interim + (m-2)] += p_prev
        else:
            # Innere Zustände: nur Nachbarn (keine Selbstschleife)
            p_right = Interim[curr, curr+1]
            if p_right > 0:
                P[src, off_interim + (curr+1)] += p_right # Nur kopieren falls > 0, weil Standardwert der Matrix 0 ist. 
            p_left = Interim[curr, curr-1]
            if p_left > 0:
                P[src, off_interim + (curr-1)] += p_left # Nur kopieren falls > 0, weil Standardwert der Matrix 0 ist. 

    # Prüfung der Zeilensummen 
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Zeile summiert sich nicht zu 1: {row_sums}"
    return P

def create_markov_matrix(n: int) -> np.ndarray:
    """
    Erstellt eine (n+1) x (n+1) Übergangs Matrix.
    Der äußere punkt (index n) ist nicht absorbierend – er kann in den letzten inneren Punkt.
    """
    #n+1 weil wir n innere Zustände haben plus einen äußeren Zustand, der nicht absorbierend ist, sondern Übergänge zur Interim-Matrix ermöglicht.
    size = n + 1
    M = np.zeros((size, size))

    
    # Innere Zustände: Übergänge zu sich selbst und zum nächsten inneren Zustand
    if n > 1:
        for i in range(n - 1):
            M[i, :n] = 1.0 / n
    # Letzter innerer Zustand: Übergänge zu sich selbst und zum äußeren Zustand, der dann Übergänge zum Interim ermöglicht
    p_outer = 1.0 / (n + 1) + 1.0 / (n + 1) ** 2
    p_inner = 1.0 / n
    M[n - 1, :n] = p_inner # Übergänge zu den inneren Zuständen immer noch gleichmäßig verteilt, aber zusätzlich Übergang zum äußeren Zustand mit p_outer
    M[n - 1, n] = p_outer # Übergang zum äußeren Zustand, der dann Übergänge zum Interim ermöglicht
    M[n - 1] /= M[n - 1].sum() # Normalisieren, damit die Zeilensumme 1 ergibt

    # Äußerer Zustand: Übergang zurück zum letzten inneren Zustand und zu sich selbst, gewichtet sodass es eher Richtung innere Zustände geht und nicht Richtung Interim
    M[n, n] = 1.0 / (n + 1) ** 2 
    M[n, n - 1] = 1.0 - M[n, n]
    return M

# Für Modell 2
def build_uniform_interim_matrix(m: int) -> np.ndarray:
    """
    Erzeugt die Interim-Übergangsmatrix der Größe m x m gemäß der
    Birth-Death-Vorschrift:
    - Zustand 0 (Index 0) <-> Zustand 1 in Formel: Selbstschleife (m-1)/m,
      Schritt nach rechts 1/m.
    - Zustand m-1 (Index m-1) <-> Zustand m: Selbstschleife (m-1)/m,
      Schritt nach links 1/m.
    - Für 1 <= i <= m-2: Übergang i -> i+1 mit (i+1)/m,
      i -> i-1 mit (m-(i+1))/m. (Da i intern 0‑basiert, ist Zustandsnummer = i+1)
    """
    P = np.zeros((m, m))
    for i in range(m):
        if i == 0:
            # Zustand 1 (Index 0)
            P[0, 0] = (m - 1) / m
            P[0, 1] = 1 / m
        elif i == m - 1:
            # Zustand m (Index m-1)
            P[m-1, m-1] = (m - 1) / m
            P[m-1, m-2] = 1 / m
        else:
            # Innere Zustände i (1..m-2) -> Zustandsnummer = i+1
            p_right = (i + 1) / m       # nach i+1 (Index i+1)
            p_left  = (m - (i + 1)) / m # nach i-1 (Index i-1)
            P[i, i+1] = p_right
            P[i, i-1] = p_left
    return P

#Für Modell 3
def build_weighted_interim_matrix(z):
    """
    Erzeugt die Übergangsmatrix P der Größe z×z nach der Vorschrift:
    - Erste Zeile: P(1,1) = (z-1)/z, P(1,2) = 1/z, sonst 0.
    - Für i = 2 .. z/2:
        P(i, i-1) = (z - (i-1)) / z
        P(i, i+1) = (i-1) / z
        Diagonale = 0.
    - Für i = z/2+1 .. z wird die Matrix durch Spiegelung an der Anti-Diagonalen
      aus den ersten z/2 Zeilen aufgebaut:
        P(i, j) = P(z+1-i, z+1-j)
    Die Matrix ist zeilenstochastisch und symmetrisch bzgl. der Anti-Diagonalen.

    """
    # Only works for even z >= 2 to ensure proper structure and avoid issues with the anti-diagonal mirroring
    if z % 2 != 0:
        raise ValueError("z muss gerade sein, um die exakte Spiegelung zu ermöglichen.")
    if z < 2:
        raise ValueError("z muss mindestens 2 sein.")

    # zxz Matrix mit Nullen initialisieren
    P = np.zeros((z, z))

    # 1. Zeile (i = 1, 1-basiert)
    P[0, 0] = (z - 1) / z
    P[0, 1] = 1 / z
    # Alle anderen Einträge in Zeile 1 sind bereits 0

    # 2. Obere Hälfte für i = 2 .. z/2 (1-basiert)
    half = z // 2
    
    # Für i = 2 .. z/2 (1-basiert) -> i = 1 .. half-1 (0-basiert)
    for i in range(2, half + 1):          # i = 2, 3, ..., z/2
        idx = i - 1                       # 0-basierter Zeilenindex
        # Übergang zu i-1
        P[idx, idx - 1] = (z - i) / z
        # Übergang zu i+1
        P[idx, idx + 1] = i / z
        # Diagonale bleibt 0 (ist bereits 0)
    # 3. Untere Hälfte durch Spiegelung an der Anti-Diagonalen
    #    für i = z/2+1 .. z (1-basiert)
    for i in range(half + 1, z + 1):
        i_idx = i - 1                     # 0-basierter Ziel-Zeilenindex
        i_spiegel = z + 1 - i             # 1-basierter Index der gespiegelten Zeile
        spiegel_row = i_spiegel - 1       # 0-basierter Index der gespiegelten Zeile
        for j in range(1, z + 1):
            j_idx = j - 1
            j_spiegel = z + 1 - j
            # Spiegel-Spalte für die gespiegelte Zeile, Wert also einfach von der gespiegelten Zeile und gespiegelten Spalte übernehmen
            P[i_idx, j_idx] = P[spiegel_row, j_spiegel - 1]

    return P

def step_counter_stationary_distribution(P: np.ndarray, change_value: float = 1e-7) -> np.ndarray:
    """
    Berechnet die Anzahl der Schritte, die benötigt werden, um von jeder Zeile der Übergangsmatrix P zur stationären Verteilung zu konvergieren.


    """
    # Anzahl der Zustände in der Markov-Kette
    n = P.shape[0]
    stat_dist_iter = np.zeros(n)
    for i in range(n):
        pi = P[i, :].copy()
        counter = 0
        for _ in range(1000000):
            counter += 1 # Schrittzähler erhöhen
            pi_next = pi @ P # Berechnung der nächsten Verteilung durch Multiplikation mit der Übergangsmatrix
            # Überprüfen, ob die Änderung der Verteilung kleiner als der Schwellenwert ist (Konvergenz zur stationären Verteilung)
            if np.linalg.norm(pi_next - pi, ord=1) < change_value: 
                break
            pi = pi_next # Aktualisierung der aktuellen Verteilung für den nächsten Schritt
        stat_dist_iter[i] = counter # Anzahl der Schritte bis zur Konvergenz für die i-te Zeile speichern
    return stat_dist_iter

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Computes the stationary distribution of a transition matrix P.
    """
    
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
   
    # Find the index of the eigenvalue that is 1 (or very close to 1)
    idx = np.argmin(np.abs(eigenvalues - 1))
    # set pi to the corresponding eigenvector, take real part and normalize to sum to 1
    pi = np.real(eigenvectors[:, idx])
    pi /= pi.sum()  # Normalize to sum to 1
    return pi

# Berechnung der vollständigen Hitting-Time-Matrix
def hitting_time_matrix(P: np.ndarray) -> np.ndarray:
    # Anzahl der Zustände in der Markov-Kette
    n = P.shape[0]
    # Initialisierung der Hitting-Time-Matrix mit Nullen
    H = np.zeros((n, n))
    # Berechnung der erwarteten Treffzeit für jedes Paar von Start- und Zielzuständen
    for start in range(n):
        for target in range(n):
            H[start, target] = expected_hitting_time(P, start, target)
    return H


def expected_hitting_time(P, start, target):
    n = P.shape[0]

    if start == target:
        return 0.0

    # We solve: h[i] = 1 + sum_j P[i,j] h[j], with h[target]=0

    A = np.eye(n)
    b = np.ones(n)

    for i in range(n):
        A[i, target] = 0.0  # remove dependency on target
        if i == target:
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        else:
            # move target contribution to RHS
            b[i] += P[i, target] * 0.0  # (kept for clarity)

    # rewrite full system properly:
    for i in range(n):
        for j in range(n):
            if j != target:
                A[i, j] -= P[i, j]

    A[target, :] = 0.0
    A[target, target] = 1.0
    b[target] = 0.0

    h = np.linalg.solve(A, b)
    return float(h[start])
if __name__ == "__main__":
    # Definition der drei Matrixgrößen für die erweiterte Markov-Ketten - n1 für Matrix 1, n2 für Matrix 2, m für die Interim-Matrix
    n1 = 4   
    n2 = 4
    m = 6 
    

    # Version 1 with uniform interim matrix
    #P = build_extended_markov_chain(n1, n2, m) #Bsp 1
    
    # Version 2 with weighted interim matrix
    P= build_extended_markov_chain_v2(n1, n2, m) #Bsp 2 
    
    
    
    eigenwerte = np.linalg.eigvals(P)
    #print eigenwerte sorted by absolute value, with 2 decimal places
    print("Eigenvalues of P (sorted by absolute value, with 2 decimal places):")
    sorted_indices = np.argsort(np.abs(eigenwerte))[::-1]
    for idx in sorted_indices:
        print(f"{eigenwerte[idx]:.8f}")
    
    
    
    print("--------------------------------")
    print("Full transition matrix P (with 2 decimal places):")
    print(P)
    print("--------------------------------")
    
    
    # Hitting-Time-Matrix
    H = hitting_time_matrix(P)
    print("Hitting-Time-Matrix H (mit 2 Dezimalstellen):")
    print(H)
    print("--------------------------------")
    
    # Cover-Time-Vector(Highest value in each row of H)
    cover_time_vector = np.max(H, axis=1)
    print("Cover-Time-Vector (mit 2 Dezimalstellen):")
    print(cover_time_vector)
    print("--------------------------------")
    #Schrittweise Berechnung der stationären Verteilung durch Iteration und Abbruchbedingung, wenn die Änderung kleiner als change_value ist. Vergleich mit der stationären Verteilung aus der Eigenvektormethode.
    stat_dist_iter = step_counter_stationary_distribution(P, change_value=0.000000000000001)
    print("Schrittzahl der stationären Verteilung durch Iteration und Vergleich mit der stationären Verteilung aus der Eigenvektormethode (mit 2 Dezimalstellen):")
    print (stat_dist_iter)
    print("--------------------------------")
    #stationäre Verteilung mittels Eigenvektormethode
    stationary_dist = stationary_distribution(P)
    print("Stationäre Verteilung:" + f" {stationary_dist}")

    P_size = []
    mixing_times = []
    for i in range(2,1000000,2): 
        
        P = build_extended_markov_chain(i,i+2,i)
        if P.shape[0] > 100:
            print(P.shape[0])
            break
        eigenwerte = np.linalg.eigvals(P)
        #print eigenwerte sorted by absolute value, with 2 decimal places
        sorted_indices = np.argsort(np.abs(eigenwerte))[::-1]
        
        second_largest = eigenwerte[sorted_indices[1]]
        #Spektraldistanz und Mischzeitannäherung 
        mixing_time = (np.log(P.shape[0])/(1-second_largest))
        # add to list 
        P_size.append(P.shape[0])
        mixing_times.append(np.abs(np.real(mixing_time)))
    counter = 0 
    for p in P_size: 
        print(f"{p},{mixing_times[counter]}")
        counter+=1
    
    df = pd.DataFrame({'P_size': P_size, 'mixing_times': mixing_times})

    # Create the line plot
    sns.lineplot(data=df, x='P_size', y='mixing_times', marker='o')  # marker adds points

    # Add labels and title
    plt.xlabel('Größe der Gesamtmatrix')
    plt.ylabel('Mischzeit in $10^6$ Schritte')
    plt.title("Annäherung an die Mischzeit bei Modell 2")

    # Show the plot
    plt.savefig("v1.pdf", format= "pdf", bbox_inches="tight")
    plt.show()
    
    
    P_size = []
    mixing_times = []
    for i in range(2,1000000,2): 
        
        P = build_extended_markov_chain_v2(i,i+2,i)
        if P.shape[0] > 100:
            break
        eigenwerte = np.linalg.eigvals(P)
        #print eigenwerte sorted by absolute value, with 2 decimal places
        sorted_indices = np.argsort(np.abs(eigenwerte))[::-1]
        
        second_largest = eigenwerte[sorted_indices[1]]
        #Spektraldistanz und Mischzeitannäherung 
        mixing_time = (np.log(P.shape[0])/(1-second_largest))
        # add to list 
        P_size.append(P.shape[0])
        mixing_times.append(np.abs(np.real(mixing_time)))
    counter = 0 
    for p in P_size: 
        print(f"{p},{mixing_times[counter]}")
        counter+=1
    
    df = pd.DataFrame({'P_size': P_size, 'mixing_times': mixing_times})

    # Create the line plot
    sns.lineplot(data=df, x='P_size', y='mixing_times', marker='o')  # marker adds points

    # Add labels and title
    plt.xlabel('Größe der Gesamtmatrix')
    plt.ylabel('Mischzeit in $10^{16}$ Schritte')
    plt.title("Annäherung an die Mischzeit bei Modell 3")

    # Show the plot
    plt.savefig("v2.pdf", format= "pdf", bbox_inches="tight")
    plt.show()



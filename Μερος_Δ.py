import numpy as np
from scipy.ndimage import zoom
import math
import time
import matplotlib.pyplot as plt
# Μέρος Α

#   Α1


W, H = 352, 288  # Βασικές παραμέτροι της εικόνας (CIF)
frame_size_4_2_0 = int(1.5 * W * H)# 152.064 bytes για YUV 4:2:0
frame_size_4_2_2 = int(2 * W * H)  # 202.752  bytes για YUV 4:2:2
frame_size_4_4_4 = int(3 * W * H)  # 304.128  bytes για YUV 4:4:4
raw_data = np.fromfile("mobile_cif.yuv", dtype=np.uint8)
bit_rate = (frame_size_4_2_0 * 30 * 8 )*(1/1000000) # 36.495.360 bits/sec frame x number_of_frames x numbers_of_bits_to_equal_byte/10^-6 ,bit rate για 4:2:0 για 30 fps


f0 = raw_data[0:frame_size_4_2_0] # Φτιάχνω εναν μονοδιαστατο πινακα που περιεχει τα bytes του πρωτου frame

y_size = W * H  # το μεγεθως των bytes που ανιστοιχουν στο κανάλι φωτεινότητας Y
uv_size = (W // 2) * (H // 2) # το μεγεθως των bytes που ανιστοιχουν σε κάθε επίπεδο χρώματος U = V

#Α2

Y_orig = f0[0:y_size].reshape((H, W))
U_orig = f0[y_size : y_size + uv_size].reshape((H // 2, W // 2))  # περνουμε τα bytes απο το f0 που αντηστοιχουν μονο στο U και τα εκφραζουμε σε H // 2 Χ W // 2 πινακα
V_orig = f0[y_size + uv_size :].reshape((H // 2, W // 2))  # περνουμε τα bytes απο το f0 που αντηστοιχουν μονο στο V και τα εκφραζουμε σε H // 2 Χ W // 2 πινακα
# Υπερδειγματοληψία (4:2:0 -> 4:4:4) με μεθοδο Bilinear interpolation 
U_4_4_4 = zoom(U_orig, 2, order=1)
V_4_4_4 = zoom(V_orig, 2, order=1)
#  Υποδειγματοληψία (4:4:4 -> 4:2:0) με μεθοδο μέσο όρο
u_f = U_4_4_4.astype(np.float64)                                                      # ΑΙ used
U_4_2_0 = np.round(
    0.25 * (
        u_f[0::2, 0::2] +   # Πάνω-αριστερά pixel κάθε 2x2
        u_f[0::2, 1::2] +   # Πάνω-δεξιά pixel κάθε 2x2
        u_f[1::2, 0::2] +   # Κάτω-αριστερά pixel κάθε 2x2
        u_f[1::2, 1::2]     # Κάτω-δεξιά pixel κάθε 2x2
    )
).astype(np.uint8)

v_f = V_4_4_4.astype(np.float64)
V_4_2_0 = np.round(
    0.25 * (
        v_f[0::2, 0::2] +
        v_f[0::2, 1::2] +
        v_f[1::2, 0::2] +
        v_f[1::2, 1::2]
    )
).astype(np.uint8)


# 1. Υπολογισμός PSNR για το κανάλι U
diff_U = U_orig.astype(np.float64) - U_4_2_0.astype(np.float64)
mse_U = np.mean(diff_U ** 2)

if mse_U == 0:
    psnr_U = math.inf
else:
    psnr_U = 10.0 * np.log10((255.0 ** 2) / mse_U)



# 1. Υπολογισμός PSNR για το κανάλι V
diff_V = V_orig.astype(np.float64) - V_4_2_0.astype(np.float64)
mse_V = np.mean(diff_V ** 2)

if mse_V == 0:
    psnr_V = math.inf
else:
    psnr_V = 10.0 * np.log10((255.0 ** 2) / mse_V)



# ΜΕΡΟΣ Β

print_intra = 0


num_frames = len(raw_data) // frame_size_4_2_0 # Βρίσκω τον αριθμό των frames
raw_frames = raw_data.reshape(num_frames, frame_size_4_2_0) # Φτιάχνω έναν δισδιάστατο πίνακα με γραμμές τον αριθμό των frames και στείλες τα bytes τους
frames = raw_frames[:, :y_size].reshape(num_frames, H, W) # Φτιάχνω έναν τρισδιάστατο πινακα ο οποιος περιεχει μονο το Υ στοιχειο απο καθε frame
number_of_blocks = (W * H) // 16

B = 4 # Block size
Qstep = 8 # Tο βήμα του προσωρινού κβαντιστή

# Frames και blocks της original εικονας τα οποια χρησιμοποιουμαι για να υπολογησουμε το SAD
original_current_frame = frames[0]
original_block = np.zeros((4, 4), dtype=np.uint8)

# Ειναι Frames και blocks τα οποια περιεχουν τα ανακατασκευασμενα bytes της εικονας απο πρωιγουμενα frames και blocks τα οποια εχει υπολογησει ο προσωρινός κβαντιστής
reconstructed_frames_array = np.zeros((num_frames,  H, W), dtype=np.uint8)
reconstructed_frame = np.zeros((H, W), dtype=np.uint8)
reconstructed_block = np.zeros((4, 4), dtype=np.uint8)

# Εδω ειναι οι block τα οποια προβλεψαν οι διαφορες μεθοδοι για intra προβλεψη 
Vertical_predicted_block = np.zeros((4, 4), dtype=np.uint8)
Horizontal_predicted_block = np.zeros((4, 4), dtype=np.uint8)
DC_predicted_block = np.zeros((4, 4), dtype=np.uint8)
predicted_frame = np.zeros((H, W), dtype=np.uint8)

# Σε περιπτωση που ειμαστε εκτος οριων 
out_of_bounds = np.zeros((4), dtype=np.uint8)
out_of_bounds[:] = 128

# Tα γειτονικα pixels που χρησιμοποιουμαι για την προβλεψη 
top = np.zeros((4), dtype=np.uint8)
left = np.zeros((4), dtype=np.uint8)


# χαρτης ο οποιος περιεχει μεσα μια τιμη για καθε block του τρεχον frame η οποια δηλωνει ποιο prediction χρησιμοποιηθηκε
prediction_map = np.zeros((H//4, W//4), dtype='<U10')


# Αρχηκοποιηση ενεργειας
residuals_energy = 0
original_frame_energy = 0
centered_frame_energy = 0
centered_block = np.zeros((4,4), dtype=np.int32)

# Αρχηκοποιηση τιμων για worst prediction
Worst_original_block = np.zeros((4,4), dtype=np.uint8)
Worst_original_block_position = np.zeros((2), dtype=np.int32)
Worst_original_block_predictions = {"Vertical": np.zeros((4, 4), dtype=np.uint8),"Horizontal": np.zeros((4, 4), dtype=np.uint8),"DC": np.zeros((4, 4), dtype=np.uint8)}
worst_sad_frame = 0


frame_pointer = 0
print_counter = 0
# Frame 0
# Intra 4X4 
for y in range(0, H, B):
    for x in range(0, W, B):
        # mode : vertical
        original_block = original_current_frame[y:y+4, x:x+4]
        if y == 0:
            Vertical_predicted_block[:B, :B] = 128
        else:
            Vertical_predicted_block[:B, :B] = reconstructed_frame[y-1, x:x+B]

        # mode : Horizontal
        if x == 0:
            Horizontal_predicted_block[:B, :B] = 128
        else:
            Horizontal_predicted_block[:B, :B] = reconstructed_frame[y:y+B, x-1].reshape(B, 1)

        # mode : DC
        if y == 0:
            top = out_of_bounds.astype(np.int32)
        else: 
            top = reconstructed_frame[y-1, x:x+B].astype(np.int32)
        if x == 0:
            left = out_of_bounds.astype(np.int32) 
        else:
            left = reconstructed_frame[y:y+B, x-1].astype(np.int32)

        # Ενώνουμε τους 4 πάνω και τους 4 αριστερούς γείτονες
        DC = np.mean(np.concatenate((top, left)))
        DC_predicted_block[:B, :B] = DC.astype(np.uint8)
        
        
        
        # SAD
        sad_vertical = np.sum(np.abs(original_block.astype(np.int32) - Vertical_predicted_block.astype(np.int32)))
        sad_horizontal = np.sum(np.abs(original_block.astype(np.int32) - Horizontal_predicted_block.astype(np.int32)))
        sad_DC = np.sum(np.abs(original_block.astype(np.int32) - DC_predicted_block.astype(np.int32)))
        sads = {"Vertical": sad_vertical,"Horizontal": sad_horizontal,"DC": sad_DC}

        best_mode = min(sads, key=sads.get)
        prediction_map[y // 4, x // 4] = best_mode


        # Ευρεση χειροτερου prediction
        worst_sad = max(sad_DC, sad_horizontal, sad_vertical)
        if worst_sad > worst_sad_frame :
            Worst_original_block = original_block
            Worst_original_block_position[:] = y, x
            Worst_original_block_predictions["Vertical"] = Vertical_predicted_block
            Worst_original_block_predictions["Horizontal"] = Horizontal_predicted_block
            Worst_original_block_predictions["DC"] = DC_predicted_block

        

        # Predicted_frame
        if best_mode == "Vertical":
            predicted_frame[y:y+B, x:x+B] = Vertical_predicted_block
        elif best_mode == "Horizontal":
            predicted_frame[y:y+B, x:x+B] = Horizontal_predicted_block
        else :
            predicted_frame[y:y+B, x:x+B] = DC_predicted_block
        

        # Reconstructed_frame
        residual = original_block.astype(np.int32) - predicted_frame[y:y+B, x:x+B].astype(np.int32)
        K_residual = Qstep * np.sign(residual/Qstep) * np.floor(np.abs(residual/Qstep) + 0.5)
        reconstructed_block = np.clip(predicted_frame[y:y+B, x:x+B].astype(np.int32) + K_residual,0,255).astype(np.uint8)
        reconstructed_frame[y:y+4, x:x+4] = reconstructed_block


        # Μετρήσεις ενέργειας
        original_frame_energy += np.sum(original_block.astype(np.int32) ** 2)
        residuals_energy += np.sum(residual ** 2)

        mean_value = np.mean(original_block)
        centered_block[:, :] = original_block.astype(np.int32) - mean_value

        centered_frame_energy += np.sum(centered_block.astype(np.int32) ** 2)

reconstructed_frames_array[frame_pointer] = reconstructed_frame

if print_intra == 1:
    print("Residual energy of the first frame =", residuals_energy)
    print("Energy of the Original first frame =", original_frame_energy)
    print("Energy of the Original first minus the mean, frame =", centered_frame_energy)

    print("\n\n\nThe block with the worst prediction is :",Worst_original_block_position,"\n\n\nWhich contains :", Worst_original_block,"\n\n\n The predictions are :",Worst_original_block_predictions)
    print("\n\n\nPrediction Map :", prediction_map)





# Δημιουργια χάρτη για την επιλεγμένη mode κάθε block.

    # Υπολογιζω το πληθος των blocks σε καθε διασταση
num_blocks_y = H // B
num_blocks_x = W // B



   # Σχεδίαση του Original Frame 0 με διαφορετικό χρώμα ανά Intra Prediction Mode

# Υπολογισμός του κέντρου κάθε block 4x4
block_centers_x = np.arange(prediction_map.shape[1]) * 4 + 2
block_centers_y = np.arange(prediction_map.shape[0]) * 4 + 2

block_grid_x, block_grid_y = np.meshgrid(
    block_centers_x,
    block_centers_y
)

# Δημιουργία διαγράμματος
plt.figure(figsize=(12, 8))

# Εμφάνιση του αρχικού Frame 0
plt.imshow(frames[0], cmap="gray", origin="upper")

# Επιλογή blocks ανάλογα με το Prediction Mode
dc_mask = prediction_map == "DC"
horizontal_mask = prediction_map == "Horizontal"
vertical_mask = prediction_map == "Vertical"

# DC: Μπλε
plt.scatter(block_grid_x[dc_mask], block_grid_y[dc_mask], color="blue", s=5, label="DC", zorder=2)

# Horizontal: Πράσινο
plt.scatter(block_grid_x[horizontal_mask], block_grid_y[horizontal_mask], color="lime", s=5, label="Horizontal", zorder=3)

# Vertical: Κόκκινο
plt.scatter(block_grid_x[vertical_mask], block_grid_y[vertical_mask], color="red", s=5, label="Vertical", zorder=4)

# Ρυθμίσεις διαγράμματος
plt.title("Intra Prediction Modes - Original Frame 0")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.xlim(0, W)
plt.ylim(H, 0)
plt.legend()
plt.tight_layout()
plt.show()




# Μερος Γ 


loop_counter = 0
# λ values
lam_array = np.array([0, 3, 12, 48])
num_lams = 4


frame_pointer += 1
best_j_position = np.zeros((4, 2), dtype=np.int32)
predicted_frame_SAD = np.zeros((H, W), dtype=np.uint8)
predicted_frame_RDO = np.zeros((num_lams, H, W), dtype=np.uint8)
reconstructed_frame = np.zeros((H, W), dtype=np.uint8)
B = 16


# RDO αρχικοποιησεις
MVp = np.zeros((4, 2), dtype=np.int32)
L_y = np.zeros((4), dtype=np.int32)
L_x = np.zeros((4), dtype=np.int32)
search_range = 16

previous_frame = reconstructed_frames_array[frame_pointer - 1]
original_current_frame = frames[frame_pointer]

# Numer of blocks ανα αξονα
num_blocks_y = H // B
num_blocks_x = W // B

# Σε αυτους τους πινακες θα αποθηκευουμε MV για καθε block για καθε αξονα 
MV_x_FS_SAD = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)
MV_y_FS_SAD = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)


# Σε αυτων τον πινακα θα αποθηκευουμε MV για καθε block για καθε αξονα για καθε λ


MV_x_FS_RDO_lam = np.zeros((num_lams, num_blocks_y, num_blocks_x), dtype=np.int32)
MV_y_FS_RDO_lam = np.zeros((num_lams, num_blocks_y, num_blocks_x), dtype=np.int32)


# Blocks για υπολογισμο του RDO
upper_block_MV = np.zeros((2), dtype=np.int32)
left_block_MV = np.zeros((2), dtype=np.int32)
upper_right_block_MV = np.zeros((2), dtype=np.int32)
MVD = np.zeros((4, 2), dtype=np.int32)

# Πινακας για να μετραμε τα sads
sad_count = 0 




#
previous_frame_block_coordinates_x = 0
previous_frame_block_coordinates_y = 0

# Αρχη μετρηση χρονου για full search
start = time.perf_counter()


# FULL SEARCH


# Επανάληψη σε όλα τα blocks του current frame
for y in range(num_blocks_y):
    for x in range(num_blocks_x):
        loop_counter += 1
        original_block = original_current_frame[y * B : y * B + B, x * B : x * B + B]
        # Αρχικοποιω μια μεταβλητη που περιεχει το καλυτερο SAD και μια αλλη για την θεση του block
        best_sad = np.inf
        best_sad_position = np.zeros((2), dtype=np.int32)
        best_j = np.zeros((4), dtype=np.float64)
        best_j[:] = np.inf
        # Υπολογιζω το MVp
        for m in range(num_lams):
            if y > 0:
                upper_block_MV = np.array([MV_y_FS_RDO_lam[m, y - 1, x], MV_x_FS_RDO_lam[m, y - 1, x]])
            else:
                upper_block_MV = np.array([0, 0])
            
            if x > 0:
                left_block_MV = np.array([MV_y_FS_RDO_lam[m, y, x - 1], MV_x_FS_RDO_lam[m, y, x - 1]])
            else:
                left_block_MV = np.array([0, 0])
            
            if y > 0 and x < num_blocks_x - 1 :
                upper_right_block_MV = np.array([MV_y_FS_RDO_lam[m, y - 1, x + 1], MV_x_FS_RDO_lam[m, y - 1, x + 1]])
            else:
                upper_right_block_MV = np.array([0, 0]) 


            
            MVp[m] = np.round(np.median([upper_right_block_MV, upper_block_MV, left_block_MV], axis=0)).astype(np.int32)
         

        


        


 # Εξέταση όλων των πιθανών μετατοπίσεων του block του οποιου εξεταζουμε
        for dy in range(-search_range, search_range + 1):


            for m in range(num_lams):
                MVD[m, 0] = dy - MVp[m,0]
                
                if MVD[m, 0] > 0 :
                    k = 2 * MVD[m, 0] - 1
                else :
                    k = -2 * MVD[m, 0]
                
                L_y[m] = 2 * math.floor(math.log2(k + 1)) + 1
            

            for dx in range(-search_range, search_range + 1):

                # Yπολογιζω τις πραγματικες συντεταγμενες 
                previous_frame_block_coordinates_x = x * B + dx
                previous_frame_block_coordinates_y = y * B + dy
                if (previous_frame_block_coordinates_y < 0 or previous_frame_block_coordinates_x < 0 or
                    previous_frame_block_coordinates_y + B > H or previous_frame_block_coordinates_x + B > W):
                    continue

                # Υπολογιζω το block με το οποιο θα συγκρινω
                previous_block = previous_frame[previous_frame_block_coordinates_y:previous_frame_block_coordinates_y + B,
                previous_frame_block_coordinates_x : previous_frame_block_coordinates_x + B]

                # Υπολογισμός SAD
                sad = np.sum(np.abs(original_block.astype(np.int32) - previous_block.astype(np.int32)))
                sad_count += 1

                #Συγκριση SAD
                if sad < best_sad:
                    best_sad = sad
                    best_sad_position[0] = dy
                    best_sad_position[1] = dx


                # Υπολογισμος RDO , J = SAD + λ·R(MVD)
                for m in range(num_lams):
                    MVD[m, 1] = dx - MVp[m, 1]
                    
                    if MVD[m, 1] > 0 :
                        k = 2 * MVD[m, 1] - 1
                    else :
                        k = -2 * MVD[m, 1]
                    
                    L_x[m] = 2 * math.floor(math.log2(k + 1)) + 1
                    
                    
                    RDO = L_x[m] + L_y[m]
                    
                    j = sad + lam_array[m] * RDO
                    
                    
                    
                    if best_j[m] > j:
                        best_j[m] = j
                        best_j_position[m, 0] = dy
                        best_j_position[m, 1] = dx




                


        # Αποθηκευω το Motion Vector για καθε block
        MV_x_FS_SAD[y, x] =  best_sad_position[1]
        MV_y_FS_SAD[y, x] = best_sad_position[0]
        dx = MV_x_FS_SAD[y,x]
        dy = MV_y_FS_SAD[y,x]


        # Aντιστάθμιση κίνησης
        predicted_frame_SAD[y * B:y * B + B, x * B:x * B + B] = previous_frame[y * B + dy : y * B + dy + B, x * B + dx : x * B + dx + B]

        for m in range(num_lams):
            MV_x_FS_RDO_lam[m, y, x] = best_j_position[m, 1]
            MV_y_FS_RDO_lam[m, y, x] = best_j_position[m, 0]
            dx = MV_x_FS_RDO_lam[m, y, x]
            dy = MV_y_FS_RDO_lam[m, y, x]
            
            


# Υπολογησμος PSNR για Full Search
mse = np.mean((original_current_frame.astype(np.float64) - predicted_frame_SAD.astype(np.float64)) ** 2)
if mse == 0:
    psnr_FS = math.inf
else:
    psnr_FS = 10.0 * np.log10((255.0 ** 2) / mse)

# Τελος μετρησης χρονου για full search
elapsed = time.perf_counter() - start
print("Full Search\n\n\n")
print("The time full search needed to finish is :", elapsed, "seconds\n\n")
print("The amount of times the Full search calculated SAD : ", sad_count, "\n\n")
print("The PSNR for Full Search is :", psnr_FS, "\n\n")


               

# Three Step Search


# Αρχη μετρηση χρονου για Three Step Search
start = time.perf_counter()


# Blocks για υπολογισμο του RDO
upper_block_MV = np.zeros((2), dtype=np.int32)
left_block_MV = np.zeros((2), dtype=np.int32)
upper_right_block_MV = np.zeros((2), dtype=np.int32)
MVD = np.zeros((4, 2), dtype=np.int32)


# Χαρτης για το ποιο λ ηταν το ποιο βελτιστο
map_Lam_TSS = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)

best_j[:] = np.inf
center_dx = 0
center_dy = 0
# ενα κεντρο για καθε  λ
center_dx_RDO = np.zeros(num_lams, dtype=np.int32)
center_dy_RDO = np.zeros(num_lams, dtype=np.int32)

MV_x_TSS_SAD = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)
MV_y_TSS_SAD = np.zeros((num_blocks_y, num_blocks_x), dtype=np.int32)


# Σε αυτων τον πινακα θα αποθηκευουμε MV για καθε block για καθε αξονα για καθε λ
MV_x_TSS_RDO_lam = np.zeros((num_lams, num_blocks_y, num_blocks_x), dtype=np.int32)
MV_y_TSS_RDO_lam = np.zeros((num_lams, num_blocks_y, num_blocks_x), dtype=np.int32)


sad_count = 0
predicted_frame_SAD = np.zeros((H, W), dtype=np.uint8)

# Επανάληψη σε όλα τα blocks του current frame
for y in range(num_blocks_y):
    for x in range(num_blocks_x):
        original_block = original_current_frame[y * B : y * B + B, x * B : x * B + B]
        center_dx = 0
        center_dy = 0


        # Αρχικοποιω μια μεταβλητη που περιεχει το καλυτερο SAD και μια αλλη για την θεση του block
        best_sad = np.inf
        best_sad_position = np.zeros((2), dtype=np.int32)

        best_j = np.full(num_lams, np.inf)

        best_j_position = np.zeros((num_lams, 2), dtype=np.int32)
        
        # Υπολογιζω το MVp
        for m in range(num_lams):
            if y > 0:
                upper_block_MV = np.array([MV_y_TSS_RDO_lam[m, y - 1, x], MV_x_TSS_RDO_lam[m, y - 1, x]])
            else:
                upper_block_MV = np.array([0, 0])
            
            if x > 0:
                left_block_MV = np.array([MV_y_TSS_RDO_lam[m, y, x - 1], MV_x_TSS_RDO_lam[m, y, x - 1]])
            else:
                left_block_MV = np.array([0, 0])
            
            if y > 0 and x < num_blocks_x - 1 :
                upper_right_block_MV = np.array([MV_y_TSS_RDO_lam[m, y - 1, x + 1], MV_x_TSS_RDO_lam[m, y - 1, x + 1]])
            else:
                upper_right_block_MV = np.array([0, 0])
        
            MVp[m] = np.round(np.median([upper_right_block_MV, upper_block_MV, left_block_MV], axis=0)).astype(np.int32)
        

        # Λουπα για τα 3 βηματα του Three steps search για sad
        for step in [16, 8, 4]:

            # Αρχικοποιηση καλυτερης θεσης
            best_dx = center_dx
            best_dy = center_dy
            best_sad = np.inf

            # Εξεταση 9 θεσεων γυρω απο το κεντρο
            for j_1 in [-1, 0, 1]:
                for i in [-1, 0, 1]:

                    # Υπολογισμος μετατοπησης καθε αξονα
                    dx = center_dx + i * step
                    dy = center_dy + j_1 * step
                    

                    # Περιορισμος στο παραθυρο +/-16
                    if abs(dx) > search_range or abs(dy) > search_range:
                        continue

                    # Πραγματικες συντεταγμενες στο previous frame
                    previous_frame_block_coordinates_x = x * B + dx
                    previous_frame_block_coordinates_y = y * B + dy

                    # Ελεγχος οριων frame
                    if (previous_frame_block_coordinates_y < 0 or previous_frame_block_coordinates_x < 0 or
                        previous_frame_block_coordinates_y + B > H or previous_frame_block_coordinates_x + B > W):
                        continue

                    # Εξαγωγη υποψηφιου block
                    previous_block_candidate = previous_frame[
                        previous_frame_block_coordinates_y:previous_frame_block_coordinates_y + B,
                        previous_frame_block_coordinates_x:previous_frame_block_coordinates_x + B
                    ]

                    # Υπολογισμος SAD
                    sad = np.sum(
                        np.abs(
                            original_block.astype(np.int32) -
                            previous_block_candidate.astype(np.int32)
                        )
                    )

                    sad_count += 1

                    # Επιλογη καλυτερης θεσης στο τρεχον βημα
                    if sad < best_sad:
                        best_sad = sad
                        best_dx = dx
                        best_dy = dy

                    


                # Νεο κεντρο για το επομενο βημα
                center_dx = best_dx
                center_dy = best_dy


        # Λουπα για τα 3 βηματα του Three steps search για ROD
        for m in range(num_lams):

            for step in [16, 8, 4]:

                best_j_step = np.inf
                best_dx_step = center_dx_RDO[m]
                best_dy_step = center_dy_RDO[m]

                for j_1 in [-1, 0, 1]:
                    for i in [-1, 0, 1]:


                        # Υπολογισμος μετατοπησης καθε αξονα
                        dx = center_dx_RDO[m] + i * step
                        dy = center_dy_RDO[m] + j_1 * step
                        
                        
                        # Περιορισμος στο παραθυρο +/-16
                        if abs(dx) > search_range or abs(dy) > search_range:
                            continue
                        
                        # Πραγματικες συντεταγμενες στο previous frame
                        previous_frame_block_coordinates_x = x * B + dx
                        previous_frame_block_coordinates_y = y * B + dy
                        
                        # Ελεγχος οριων frame
                        if (previous_frame_block_coordinates_y < 0 or previous_frame_block_coordinates_x < 0 or
                            previous_frame_block_coordinates_y + B > H or previous_frame_block_coordinates_x + B > W):
                            continue

                        previous_block_candidate = previous_frame[
                        previous_frame_block_coordinates_y:previous_frame_block_coordinates_y + B,
                        previous_frame_block_coordinates_x:previous_frame_block_coordinates_x + B
                        ]


                        # Υπολογισμος SAD
                        sad = np.sum(
                            np.abs(
                                original_block.astype(np.int32) -
                                previous_block_candidate.astype(np.int32)
                            )
                        )
                        # Υπολογισμος MVD και RDO
                        MVD[m, 0] = dy - MVp[m,0]
                        
                        if MVD[m, 0] > 0 :
                            k = 2 * MVD[m, 0] - 1
                        else :
                            k = -2 * MVD[m, 0]
                        
                        L_y[m] = 2 * math.floor(math.log2(k + 1)) + 1
                        
                        # Υπολογισμος RDO , J = SAD + λ·R(MVD)
                        MVD[m, 1] = dx - MVp[m ,1]
                        
                        if MVD[m, 1] > 0 :
                            k = 2 * MVD[m, 1] - 1
                        else :
                            k = -2 * MVD[m, 1]
                        
                        L_x[m] = 2 * math.floor(math.log2(k + 1)) + 1
                        
                        
                        RDO = L_x[m] + L_y[m]

                        j = sad + lam_array[m] * RDO

                        if j < best_j_step:
                            best_j_step = j
                            best_dx_step = dx
                            best_dy_step = dy

                # Νεο κεντρο για το επομενο βημα
                center_dx_RDO[m] = best_dx_step
                center_dy_RDO[m] = best_dy_step

        
                       

            
            


        # Αποθηκευση τελικου Motion Vector
        MV_x_TSS_SAD[y, x] = center_dx
        MV_y_TSS_SAD[y, x] = center_dy
        dx = MV_x_TSS_SAD[y,x]
        dy = MV_y_TSS_SAD[y,x]


        # Aντιστάθμιση κίνησης
        predicted_frame_SAD[y * B:y * B + B, x * B:x * B + B] = previous_frame[y * B + dy : y * B + dy + B, x * B + dx : x * B + dx + B]

        for m in range(num_lams):
            MV_x_TSS_RDO_lam[m, y, x] = center_dx_RDO[m]
            MV_y_TSS_RDO_lam[m, y, x] = center_dy_RDO[m]
            dx = MV_x_TSS_RDO_lam[m, y, x]
            dy = MV_y_TSS_RDO_lam[m, y, x]
            
            # Aντιστάθμιση κίνησης
            predicted_frame_RDO[m, y * B:y * B + B, x * B:x * B + B] = previous_frame[y * B + dy : y * B + dy + B, x * B + dx : x * B + dx + B]
         



# Υπολογησμος PSNR για three step search
mse = np.mean((original_current_frame.astype(np.float64) -predicted_frame_SAD.astype(np.float64)) ** 2)
if mse == 0:
    psnr_TSS = math.inf
else:
    psnr_TSS = 10.0 * np.log10((255.0 ** 2) / mse)




# Τελος μετρhσης χρονου για Three Step Search
elapsed = time.perf_counter() - start
print("Three Step Search\n\n\n")
print("The time Three Step Search needed to finish is :", elapsed, "seconds\n\n")
print("The amount of times the Three Step Search calculated SAD : ", sad_count, "\n\n")
print("The PSNR for Three Step Search is :", psnr_TSS, "\n\n")













# Σχεδιάστε το πεδίο των διανυσμάτων κίνησης πάνω στο frame



    # Υπολογιζω το κεντρο καθε block
x = np.arange(num_blocks_x) * B + B / 2
y = np.arange(num_blocks_y) * B + B / 2

X, Y = np.meshgrid(x, y)

# Δημιουργία πλέγματος 2 γραμμών και 3 στηλών
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Μετατρέπουμε το 2D πλέγμα axes σε 1D λίστα για εύκολη προσπέλαση
axes = axes.flatten()





    # Εμφάνιση του αρχικού frame 0

axes[0].imshow(frames[0], cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[0].scatter(X, Y, color="red" ,s=5, zorder=2)

axes[0].set_title("Original Frame 0")
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].set_xlim(0, original_current_frame.shape[1])
axes[0].set_ylim(original_current_frame.shape[0], 0)



# Εμφάνιση του αρχικού reconstructed frame 0
axes[1].imshow(reconstructed_frames_array[0], cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[1].scatter(X, Y, color="red" ,s=5, zorder=2)

axes[1].set_title("Reconstructed Frame 0")
axes[1].set_xlabel("X (pixels)")
axes[1].set_ylabel("Y (pixels)")
axes[1].set_xlim(0, original_current_frame.shape[1])
axes[1].set_ylim(original_current_frame.shape[0], 0)


    # Εμφάνιση του αρχικού frame 1 για Full Search
axes[2].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[2].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[2].quiver(X, Y, MV_x_FS_SAD, MV_y_FS_SAD, color="red", angles="xy", scale_units="xy", scale=1)

axes[2].set_title("MV map for every block in Frame 1 with FS")
axes[2].set_xlabel("X (pixels)")
axes[2].set_ylabel("Y (pixels)")
axes[2].set_xlim(0, original_current_frame.shape[1])
axes[2].set_ylim(original_current_frame.shape[0], 0)


 # Εμφάνιση του αρχικού frame 1 για Three Step Search
axes[3].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[3].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[3].quiver(X, Y, MV_x_TSS_SAD, MV_y_TSS_SAD, color="red", angles="xy", scale_units="xy", scale=1)

axes[3].set_title("MV map for every block in Frame 1 with TSS")
axes[3].set_xlabel("X (pixels)")
axes[3].set_ylabel("Y (pixels)")
axes[3].set_xlim(0, original_current_frame.shape[1])
axes[3].set_ylim(original_current_frame.shape[0], 0)

plt.tight_layout()
plt.show()



# Eμφανιση φραφιματοςω για FS TSS με RDO κρητιριο για καθε λ


X, Y = np.meshgrid(x, y)

# Δημιουργία πλέγματος 2 γραμμών και 3 στηλών
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Μετατρέπουμε το 2D πλέγμα axes σε 1D λίστα για εύκολη προσπέλαση
axes = axes.flatten()





    # Εμφάνιση του αρχικού frame 0

axes[0].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[0].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[0].quiver(X, Y, MV_x_TSS_RDO_lam[0], MV_y_TSS_RDO_lam[0], color="red", angles="xy", scale_units="xy", scale=1)

axes[0].set_title("MV map for every block in Frame 1 with TSS with criterion RDO for λ = 0")
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].set_xlim(0, original_current_frame.shape[1])
axes[0].set_ylim(original_current_frame.shape[0], 0)



# Εμφάνιση του αρχικού reconstructed frame 0
axes[1].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[1].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[1].quiver(X, Y, MV_x_TSS_RDO_lam[1], MV_y_TSS_RDO_lam[1], color="red", angles="xy", scale_units="xy", scale=1)

axes[1].set_title("MV map for every block in Frame 1 with TSS with criterion RDO for λ = 3")
axes[1].set_xlabel("X (pixels)")
axes[1].set_ylabel("Y (pixels)")
axes[1].set_xlim(0, original_current_frame.shape[1])
axes[1].set_ylim(original_current_frame.shape[0], 0)


    # Εμφάνιση του αρχικού frame 1 για Full Search
axes[2].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[2].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[2].quiver(X, Y, MV_x_TSS_RDO_lam[2], MV_y_TSS_RDO_lam[2], color="red", angles="xy", scale_units="xy", scale=1)

axes[2].set_title("MV map for every block in Frame 1 with TSS with criterion RDO for λ = 12")
axes[2].set_xlabel("X (pixels)")
axes[2].set_ylabel("Y (pixels)")
axes[2].set_xlim(0, original_current_frame.shape[1])
axes[2].set_ylim(original_current_frame.shape[0], 0)


 # Εμφάνιση του αρχικού frame 1 για Three Step Search
axes[3].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[3].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[3].quiver(X, Y, MV_x_TSS_RDO_lam[3], MV_y_TSS_RDO_lam[3], color="red", angles="xy", scale_units="xy", scale=1)

axes[3].set_title("MV map for every block in Frame 1 with TSS with criterion RDO for λ = 48")
axes[3].set_xlabel("X (pixels)")
axes[3].set_ylabel("Y (pixels)")
axes[3].set_xlim(0, original_current_frame.shape[1])
axes[3].set_ylim(original_current_frame.shape[0], 0)

plt.tight_layout()
plt.show()


# για Full seasrch


# Δημιουργία πλέγματος 2 γραμμών και 3 στηλών
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Μετατρέπουμε το 2D πλέγμα axes σε 1D λίστα για εύκολη προσπέλαση
axes = axes.flatten()


axes[0].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[0].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[0].quiver(X, Y, MV_x_FS_RDO_lam[0], MV_y_FS_RDO_lam[0], color="red", angles="xy", scale_units="xy", scale=1)

axes[0].set_title("MV map for every block in Frame 1 with FS with criterion RDO for λ = 0")
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].set_xlim(0, original_current_frame.shape[1])
axes[0].set_ylim(original_current_frame.shape[0], 0)



# Εμφάνιση του αρχικού reconstructed frame 0
axes[1].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[1].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[1].quiver(X, Y, MV_x_FS_RDO_lam[1], MV_y_FS_RDO_lam[1], color="red", angles="xy", scale_units="xy", scale=1)

axes[1].set_title("MV map for every block in Frame 1 with FS with criterion RDO for λ = 3")
axes[1].set_xlabel("X (pixels)")
axes[1].set_ylabel("Y (pixels)")
axes[1].set_xlim(0, original_current_frame.shape[1])
axes[1].set_ylim(original_current_frame.shape[0], 0)


    # Εμφάνιση του αρχικού frame 1 για Full Search
axes[2].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[2].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[2].quiver(X, Y, MV_x_FS_RDO_lam[2], MV_y_FS_RDO_lam[2], color="red", angles="xy", scale_units="xy", scale=1)

axes[2].set_title("MV map for every block in Frame 1 with FS with criterion RDO for λ = 12")
axes[2].set_xlabel("X (pixels)")
axes[2].set_ylabel("Y (pixels)")
axes[2].set_xlim(0, original_current_frame.shape[1])
axes[2].set_ylim(original_current_frame.shape[0], 0)


 # Εμφάνιση του αρχικού frame 1 για Three Step Search
axes[3].imshow(original_current_frame, cmap="gray", origin="upper")
# Σχεδίαση των διανυσμάτων κίνησης
axes[3].scatter(X, Y, color="red" ,s=5, zorder=2)
axes[3].quiver(X, Y, MV_x_FS_RDO_lam[3], MV_y_FS_RDO_lam[3], color="red", angles="xy", scale_units="xy", scale=1)

axes[3].set_title("MV map for every block in Frame 1 with FS with criterion RDO for λ = 48")
axes[3].set_xlabel("X (pixels)")
axes[3].set_ylabel("Y (pixels)")
axes[3].set_xlim(0, original_current_frame.shape[1])
axes[3].set_ylim(original_current_frame.shape[0], 0)

plt.tight_layout()
plt.show()





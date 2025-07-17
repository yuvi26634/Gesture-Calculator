import cv2
import mediapipe as mp
import customtkinter as ctk
from threading import Thread
import time
import math
import ast
import operator


# --- SAFE CALCULATION LOGIC  ---
def safe_calculate(expression):
    try:
        if expression.endswith(('*', '/', '+', '-')):
            expression = expression[:-1]
        node = ast.parse(expression, mode='eval').body
        allowed_ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.USub: operator.neg
        }

        def eval_expr(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return allowed_ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_ops[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(node)

        return eval_expr(node)
    except (TypeError, SyntaxError, KeyError, ZeroDivisionError):
        raise ValueError("Invalid Expression")


# --- GUI SETUP ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("Gesture Calculator")
root.geometry("380x520")
buttons = {}
full_expression = ""
entry_cleared = True


# --- GUI FUNCTIONS ---
def update_full_expression_display():
    full_expression_label.configure(text=full_expression)


def update_entry_display(value):
    entry_label.configure(text=str(value))


def backspace():
    global entry_cleared
    current_text = entry_label.cget("text")
    if current_text != "0" and current_text != "Error" and not entry_cleared:
        new_text = current_text[:-1]
        if not new_text:
            update_entry_display("0")
            entry_cleared = True
        else:
            update_entry_display(new_text)


def button_press(symbol):
    global full_expression, entry_cleared
    if str(symbol) in "0123456789":
        current_text = entry_label.cget("text")
        if current_text == "0" or entry_cleared:
            update_entry_display(symbol)
            entry_cleared = False
        else:
            update_entry_display(current_text + str(symbol))
    elif symbol == ".":
        current_text = entry_label.cget("text")
        if "." not in current_text and current_text != "Error":
            update_entry_display(current_text + ".")
            entry_cleared = False
    elif str(symbol) in "+-*/":
        current_text = entry_label.cget("text")
        if not full_expression.endswith(('* ', '/ ', '+ ', '- ')):
            full_expression += str(current_text) + f" {str(symbol)} "
        update_full_expression_display()
        entry_cleared = True
    elif symbol == "C":
        clear_display()
    elif symbol == "=":
        calculate_result()
    elif symbol == "⌫":
        backspace()
    if str(symbol) in buttons: highlight_button(buttons[str(symbol)])


def clear_display():
    global full_expression, entry_cleared
    full_expression = "";
    update_full_expression_display()
    update_entry_display("0");
    entry_cleared = True


def calculate_result():
    global full_expression, entry_cleared
    current_text = entry_label.cget("text")
    full_expression += str(current_text)
    try:
        result = safe_calculate(full_expression.replace(" ", ""))
        update_entry_display(round(result, 10))
    except ValueError:
        update_entry_display("Error")
    full_expression = "";
    update_full_expression_display()
    entry_cleared = True


def highlight_button(button):
    button.configure(fg_color="#00BF63")
    root.after(200, lambda: button.configure(fg_color=button.original_color))


# --- GUI LAYOUT ---
display_frame = ctk.CTkFrame(root, corner_radius=0);
display_frame.pack(fill="both", expand=True)
full_expression_label = ctk.CTkLabel(display_frame, text="", font=("Arial", 20), anchor="e");
full_expression_label.pack(fill="x", padx=10, pady=(10, 0))
entry_label = ctk.CTkLabel(display_frame, text="0", font=("Arial", 60, "bold"), anchor="e");
entry_label.pack(fill="x", padx=10, pady=(0, 10))
button_frame = ctk.CTkFrame(root, corner_radius=0);
button_frame.pack(fill="both", expand=True, side="bottom")
button_layout = [
    ('C', 1, 0), ('⌫', 1, 1), ('/', 1, 3),
    ('7', 2, 0), ('8', 2, 1), ('9', 2, 2), ('*', 2, 3),
    ('4', 3, 0), ('5', 3, 1), ('6', 3, 2), ('-', 3, 3),
    ('1', 4, 0), ('2', 4, 1), ('3', 4, 2), ('+', 4, 3),
    ('0', 5, 0, 2), ('.', 5, 2), ('=', 5, 3)
]
for i in range(1, 6): button_frame.grid_rowconfigure(i, weight=1)
for i in range(4): button_frame.grid_columnconfigure(i, weight=1)
for (text, row, col, *span) in button_layout:
    colspan = span[0] if span else 1
    if text in "+-*/=":
        color = "#FF9500"
    elif text in ('C', '⌫'):
        color = "#D4D4D2"
    else:
        color = "#505050"
    button = ctk.CTkButton(
        button_frame, text=text, font=("Arial", 24),
        command=lambda t=text: button_press(t), fg_color=color
    )
    button.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=5, pady=5)
    button.original_color = color;
    buttons[text] = button


# --- NEW: INSTRUCTIONS WINDOW ---
def create_instructions_window():
    instructions_window = ctk.CTkToplevel(root)
    instructions_window.title("Gesture Instructions")
    instructions_window.geometry("450x350")
    instructions_window.transient(root)  # Keep on top of the main window

    ctk.CTkLabel(instructions_window, text="Gesture Guide", font=("Arial", 20, "bold")).pack(pady=10)

    # Use a frame and grid for clean layout
    frame = ctk.CTkFrame(instructions_window)
    frame.pack(pady=10, padx=20, fill="both", expand=True)

    gestures = {
        "👍": "Add (+): Thumbs Up",
        "👎": "Decimal (.): Thumbs Down",
        "✊ (pinky)": "Subtract (-): Pinky Up",
        "🤟": "Multiply (*): Rock On",
        "👆+👆": "Divide (/): Two Index Fingers",
        "👌": "Equals (=): OK Sign",
        "🤙": "Clear (C): Call Me Sign",
        "👍 (L-shape)": "Backspace (⌫): L-Shape"
    }

    for i, (icon, desc) in enumerate(gestures.items()):
        icon_label = ctk.CTkLabel(frame, text=icon, font=("Segoe UI Emoji", 24))
        icon_label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        desc_label = ctk.CTkLabel(frame, text=desc, font=("Arial", 14))
        desc_label.grid(row=i, column=1, padx=10, pady=5, sticky="w")


# --- MEDIAPIPE AND GESTURE LOGIC ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def get_finger_orientation(landmarks, tip_id, base_id):
    tip = landmarks[tip_id];
    base = landmarks[base_id]
    dx = abs(tip.x - base.x);
    dy = abs(tip.y - base.y)
    if dy > dx * 1.5: return "vertical"
    if dx > dy * 1.5: return "horizontal"
    return "diagonal"


def camera_loop():
    cap = cv2.VideoCapture(0)
    last_detected_symbol, gesture_start_time = None, 0
    action_taken_this_session = False
    GESTURE_HOLD_TIME = 1.0

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_symbol, hand_present_now = None, bool(results.multi_hand_landmarks)

        if hand_present_now:
            # Draw landmarks on the clean camera feed
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not action_taken_this_session:
                num_hands = len(results.multi_hand_landmarks)
                all_hands_fingers, total_fingers = [], 0
                for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_type.classification[0].label
                    fingers = []
                    if hand_label == "Right":
                        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
                    else:
                        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
                    for tip_id in [8, 12, 16, 20]:
                        fingers.append(
                            1 if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y else 0)
                    all_hands_fingers.append(fingers)
                    total_fingers += fingers.count(1)

                if num_hands == 2 and all(f == [0, 1, 0, 0, 0] for f in all_hands_fingers):
                    current_symbol = '/'
                elif num_hands == 1:
                    fingers_hand1 = all_hands_fingers[0]
                    hand1_lms = results.multi_hand_landmarks[0].landmark
                    thumb_tip, index_tip, thumb_base = hand1_lms[4], hand1_lms[8], hand1_lms[2]
                    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

                    if dist < 0.05:
                        current_symbol = '='
                    elif fingers_hand1 == [1, 0, 0, 0, 0]:
                        current_symbol = '+' if thumb_tip.y < thumb_base.y else '.'
                    elif fingers_hand1 == [0, 0, 0, 0, 1]:
                        current_symbol = '-'
                    elif fingers_hand1 == [1, 0, 0, 0, 1]:
                        current_symbol = 'C'
                    elif fingers_hand1 == [1, 1, 0, 0, 1]:
                        current_symbol = '*'
                    elif fingers_hand1 == [1, 1, 0, 0, 0]:
                        current_symbol = '⌫'

                if current_symbol is None and total_fingers <= 9:
                    current_symbol = str(total_fingers)

                if current_symbol is not None:
                    if current_symbol != last_detected_symbol:
                        last_detected_symbol, gesture_start_time = current_symbol, time.time()
                    else:
                        elapsed_time = time.time() - gesture_start_time
                        progress = min(elapsed_time / GESTURE_HOLD_TIME, 1.0)
                        bar_w = int(progress * 200)
                        cv2.rectangle(img, (10, h - 30), (210, h - 10), (50, 50, 50), 3)
                        cv2.rectangle(img, (10, h - 30), (10 + bar_w, h - 10), (0, 255, 0), -1)
                        if progress >= 1.0:
                            button_press(last_detected_symbol)
                            action_taken_this_session = True
        else:
            if action_taken_this_session:
                print("Hand removed. Ready for next input.")
            action_taken_this_session, last_detected_symbol, gesture_start_time = False, None, 0

        cv2.imshow("Gesture Cam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()


# --- SCRIPT START ---
camera_thread = Thread(target=camera_loop, daemon=True)
camera_thread.start()

# NEW: Create the instructions window when the app starts
create_instructions_window()

# Launch GUI
root.mainloop()
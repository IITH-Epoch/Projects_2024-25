import streamlit as st
import matplotlib.pyplot as plt
from expn_infer import resolve_symbols_on_img, predict_latex_expression, latex_to_sympy
import re
import os
from PIL import Image
import io
import sympy
from sympy import symbols, sympify, solve, Eq, simplify, N, E

def solve_with_streamlit(latex_expr):
    """Modified version of solve_or_evaluate for Streamlit"""
    normalized_expr = latex_to_sympy(latex_expr)
    st.text(f"Normalized Expression: {normalized_expr}")

    if "=" in normalized_expr:
        lhs, rhs = normalized_expr.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Create the equation
        equation = Eq(sympify(lhs), sympify(rhs))
        st.text(f"SymPy Expression: {equation}")

        # Identify variables in the equation
        variables = list(set(sympy.Symbol(sym) for sym in re.findall(r'[a-zA-Z]+', normalized_expr) if sym not in ['E', 'pi']))

        if not variables:
            st.warning("No variables found in the equation.")
            return

        # If there's only one variable, solve for it
        if len(variables) == 1:
            solutions = solve(equation, variables[0])
            numerical_solutions = [N(sol, 10) for sol in solutions]
            st.success(f"Solutions for {variables[0]}: {numerical_solutions}")
        else:
            st.info(f"Multiple variables found: {variables}")
            # Create a dropdown for variable selection
            var_to_solve = st.selectbox(
                "Select variable to solve for:",
                [str(var) for var in variables]
            )
            var_symbol = sympy.Symbol(var_to_solve)
            if var_symbol in variables:
                solutions = solve(equation, var_symbol)
                numerical_solutions = [N(sol, 10) for sol in solutions]
                st.success(f"Solutions for {var_symbol}: {numerical_solutions}")
            else:
                st.error(f"Variable {var_to_solve} not found in the equation.")

        # Simplify the equation
        simplified_eq = simplify(equation)
        st.text(f"Simplified equation: {simplified_eq}")

    else:
        # If it's just an expression, evaluate it
        expr = sympify(normalized_expr)
        simplified_expr = simplify(expr)
        st.text(f"Simplified expression: {simplified_expr}")
        
        # Try to evaluate numerically
        try:
            numerical_value = N(simplified_expr, 10)
            st.success(f"Numerical value: {numerical_value}")
        except:
            st.warning("Unable to evaluate numerically. The expression may contain variables.")

def main():
    st.title("Handwritten Mathematical Equation Solver")
    st.write("Upload an image of a handwritten mathematical equation to get started!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a temporary directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_path = os.path.join('temp', "temp_image.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Process the image
            symbols, _, stacked_list, script_levels, _, ax = resolve_symbols_on_img(temp_path)
            
            # Display the processed image with bounding boxes
            st.write("### Processed Image with Symbol Detection")
            fig = ax.figure
            st.pyplot(fig)
            plt.close(fig)

            # Make prediction
            model_path = 'final_math_symbol_cnn.pth'
            latex_expr = predict_latex_expression(symbols, script_levels, stacked_list, model_path)
            latex_expr = re.sub(r"\\frac(?!{)", "-", latex_expr)

            # Display the predicted LaTeX
            st.write("### Predicted LaTeX Expression")
            st.code(latex_expr, language="latex")

            # Allow user to edit the LaTeX
            st.write("### Edit LaTeX (if needed)")
            corrected_latex = st.text_input("Edit LaTeX expression", value=latex_expr)

            # Display the rendered equation
            st.write("### Rendered Equation")
            st.latex(corrected_latex)

            # Add solve/simplify option
            if st.button("Solve/Simplify Expression"):
                st.write("### Solution")
                try:
                    with st.spinner("Solving..."):
                        solve_with_streamlit(corrected_latex)
                except Exception as e:
                    st.error(f"Error solving equation: {str(e)}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    st.set_page_config(page_title="Math Equation Solver", layout="wide")
    main()
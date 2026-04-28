"""
Few-shot chain-of-thought exemplars for GSM8K and MATH evaluation.

GSM8K: 8-shot, standard exemplars (Cobbe et al., 2021).
MATH: 4-shot, from official Qwen2.5-Math evaluation (examples.py).
Separator: "\n\n\n" (three newlines) — matches official Qwen2.5-Math setup.
"""

from typing import Literal


_SHOT_SEP = "\n\n\n"

GSM8K_8SHOT_EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.",
    },
]

MATH_4SHOT_EXEMPLARS = [
    {
        "question": r"Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "answer": "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}",
    },
    {
        "question": r"What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "answer": "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
    },
    {
        "question": r"If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "answer": "Let's think step by step\nIf $(x,y)$ lies on the circle, so does $(x,-y),$ $(-x,-y),$ and $(-x,y),$ (which all give the same value of $|x| + |y|$), so we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$ Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence, $1 + 2xy \\le 2,$ which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$ so the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
    },
    {
        "question": r"If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "answer": "Let's think step by step\nThe condition $f(f(x))=x$ means that $f$ is the inverse of itself, so its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$, and therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
    },
]


def build_gsm8k_prompt(problem: str) -> str:
    """Build 8-shot GSM8K prompt using official Qwen2.5-Math format."""
    parts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in GSM8K_8SHOT_EXEMPLARS]
    prefix = _SHOT_SEP.join(parts)
    return f"{prefix}{_SHOT_SEP}Question: {problem}\nAnswer: "


def build_math_prompt(problem: str) -> str:
    """Build 4-shot MATH prompt using official Qwen2.5-Math format."""
    parts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in MATH_4SHOT_EXEMPLARS]
    prefix = _SHOT_SEP.join(parts)
    return f"{prefix}{_SHOT_SEP}Question: {problem}\nAnswer: "


def build_few_shots_prompt(problem: str, problem_source: Literal["gsm8k", "math", "augmented_math", "augmented_gsm8k"]) -> str:
    """Build few-shots prompt based on problem source."""
    if problem_source in ["gsm8k", "augmented_gsm8k"]:
        return build_gsm8k_prompt(problem)
    elif problem_source in ["math", "augmented_math"]:
        return build_math_prompt(problem)
    else:
        raise ValueError(f"Unknown problem source: {problem_source}")


def build_zero_shot_prompt(problem: str, problem_source = None) -> str:
    """Standard 0-shot prompt matching training format."""
    return f"Question: {problem}\nAnswer: "

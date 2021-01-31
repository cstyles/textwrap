use crate::core::Fragment;
use std::cell::RefCell;

/// Cache for line numbers. This is necessary to avoid a O(n**2)
/// behavior when computing line numbers in [`wrap_optimal_fit`].
struct LineNumbers {
    line_numbers: RefCell<Vec<usize>>,
}

impl LineNumbers {
    fn new(size: usize) -> Self {
        let mut line_numbers = Vec::with_capacity(size);
        line_numbers.push(0);
        LineNumbers {
            line_numbers: RefCell::new(line_numbers),
        }
    }

    fn get<T>(&self, i: usize, minima: &[(usize, T)]) -> usize {
        while self.line_numbers.borrow_mut().len() < i + 1 {
            let pos = self.line_numbers.borrow().len();
            let line_number = 1 + self.get(minima[pos].0, &minima);
            self.line_numbers.borrow_mut().push(line_number);
        }

        self.line_numbers.borrow()[i]
    }
}

/// Per-line penalty. This is added for every line, which makes it
/// expensive to output more lines than the minimum required.
const NLINE_PENALTY: f64 = 1000.0;

/// Penalty given to a line with the maximum possible gap, i.e., a
/// line with a width of zero.
const MAX_LINE_PENALTY: f64 = 10000.0;

/// Per-character cost for lines that overflow the target line width.
const OVERFLOW_PENALTY: f64 = 2.0 * MAX_LINE_PENALTY;

/// The last line is short if it is less than 1/4 of the target width.
const SHORT_LINE_FRACTION: usize = 4;

/// Penalize a short last line.
const SHORT_LAST_LINE_PENALTY: f64 = 125.0;

/// Penalty for lines ending with a hyphen.
const HYPHEN_PENALTY: f64 = 150.0;

/// Compute the cost of the line containing `fragments[i..j]` given a
/// pre-computed `line_width` and `target_width`. The optimal cost of
/// breaking fragments[..i] into lines is given by `minimum_cost`.
fn line_penalty<'a, F: Fragment>(
    (i, j): (usize, usize),
    fragments: &'a [F],
    line_width: usize,
    target_width: usize,
    minimum_cost: f64,
) -> f64 {
    // Each new line costs NLINE_PENALTY. This prevents creating more
    // lines than necessary.
    let mut cost = minimum_cost + NLINE_PENALTY;

    // Next, we add a penalty depending on the line length.
    if line_width > target_width {
        // Lines that overflow get a hefty penalty.
        let overflow = line_width - target_width;
        cost += overflow as f64 * OVERFLOW_PENALTY;
    } else if j < fragments.len() {
        // Other lines (except for the last line) get a milder penalty
        // which increases quadratically from 0 to MAX_LINE_PENALTY.
        let gap = (target_width - line_width) as f64 / target_width as f64;
        cost += gap * gap * MAX_LINE_PENALTY;
    } else if i + 1 == j && line_width < target_width / SHORT_LINE_FRACTION {
        // The last line can have any size gap, but we do add a
        // penalty if the line is very short (typically because it
        // contains just a single word).
        cost += SHORT_LAST_LINE_PENALTY;
    }

    // Finally, we discourage hyphens.
    if fragments[j - 1].penalty_width() > 0 {
        // TODO: this should use a penalty value from the fragment
        // instead.
        cost += HYPHEN_PENALTY;
    }

    cost
}

/// Wrap abstract fragments into lines with an optimal-fit algorithm.
///
/// The `line_widths` map line numbers (starting from 0) to a target
/// line width. This can be used to implement hanging indentation.
///
/// The fragments must already have been split into the desired
/// widths, this function will not (and cannot) attempt to split them
/// further when arranging them into lines.
///
/// # Optimal-Fit Algorithm
///
/// The algorithm considers all possible break points and picks the
/// breaks which minimizes the gaps at the end of each line. More
/// precisely, the algorithm assigns a cost or penalty to each break
/// point, determined by `cost = gap * gap` where `gap = target_width -
/// line_width`. Shorter lines are thus penalized more heavily since
/// they leave behind a larger gap.
///
/// We can illustrate this with the text “To be, or not to be: that is
/// the question”. We will be wrapping it in a narrow column with room
/// for only 10 characters. The [greedy
/// algorithm](super::wrap_first_fit) will produce these lines, each
/// annotated with the corresponding penalty:
///
/// ```text
/// "To be, or"   1² =  1
/// "not to be:"  0² =  0
/// "that is"     3² =  9
/// "the"         7² = 49
/// "question"    2² =  4
/// ```
///
/// We see that line four with “the” leaves a gap of 7 columns, which
/// gives it a penalty of 49. The sum of the penalties is 63.
///
/// There are 10 words, which means that there are `2_u32.pow(9)` or
/// 512 different ways to typeset it. We can compute
/// the sum of the penalties for each possible line break and search
/// for the one with the lowest sum:
///
/// ```text
/// "To be,"     4² = 16
/// "or not to"  1² =  1
/// "be: that"   2² =  4
/// "is the"     4² = 16
/// "question"   2² =  4
/// ```
///
/// The sum of the penalties is 41, which is better than what the
/// greedy algorithm produced.
///
/// Searching through all possible combinations would normally be
/// prohibitively slow. However, it turns out that the problem can be
/// formulated as the task of finding column minima in a cost matrix.
/// This matrix has a special form (totally monotone) which lets us
/// use a [linear-time algorithm called
/// SMAWK](https://lib.rs/crates/smawk) to find the optimal break
/// points.
///
/// This means that the time complexity remains O(_n_) where _n_ is
/// the number of words. Compared to
/// [`wrap_first_fit`](super::wrap_first_fit), this function is about
/// 4 times slower.
///
/// The optimization of per-line costs over the entire paragraph is
/// inspired by the line breaking algorithm used in TeX, as described
/// in the 1981 article [_Breaking Paragraphs into
/// Lines_](http://www.eprg.org/G53DOC/pdfs/knuth-plass-breaking.pdf)
/// by Knuth and Plass. The implementation here is based on [Python
/// code by David
/// Eppstein](https://github.com/jfinkels/PADS/blob/master/pads/wrap.py).
///
/// **Note:** Only available when the `smawk` Cargo feature is
/// enabled.
pub fn wrap_optimal_fit<'a, T: Fragment, F: Fn(usize) -> usize>(
    fragments: &'a [T],
    line_widths: F,
) -> Vec<&'a [T]> {
    let mut widths = Vec::with_capacity(fragments.len() + 1);
    let mut width = 0;
    widths.push(width);
    for fragment in fragments {
        width += fragment.width() + fragment.whitespace_width();
        widths.push(width);
    }

    let line_numbers = LineNumbers::new(fragments.len());
    let minima = smawk::online_column_minima(
        0.0,
        widths.len(),
        |minima: &[(usize, f64)], i: usize, j: usize| {
            // Line number for fragment `i`.
            let line_number = line_numbers.get(i, &minima);
            let target_width = std::cmp::max(1, line_widths(line_number));

            // Compute the width of a line spanning fragments[i..j] in
            // constant time. We need to adjust widths[j] by subtracting
            // the whitespace of fragment[j-i] and then add the penalty.
            let line_width = widths[j] - widths[i] - fragments[j - 1].whitespace_width()
                + fragments[j - 1].penalty_width();

            // The line containing fragments[i..j]. We start with
            // minima[i].1, which is the optimal cost for breaking
            // before fragments[i].
            let minimum_cost = minima[i].1;
            line_penalty((i, j), fragments, line_width, target_width, minimum_cost)
        },
    );

    let mut lines = Vec::with_capacity(line_numbers.get(fragments.len(), &minima));
    let mut pos = fragments.len();
    loop {
        let prev = minima[pos].0;
        lines.push(&fragments[prev..pos]);
        pos = prev;
        if pos == 0 {
            break;
        }
    }

    lines.reverse();
    lines
}

//! Argmin/Argmax implementations using Thermite SIMD abstractions.

use thermite::{Vector, register::well_formed::WellFormedFloatElement, simd::FloatSimd};

use crate::LoadKernel;

#[inline(always)]
pub fn arg_minmax<S, T, L, const I: usize>(loader: &L, data: [&[T]; I]) -> Option<(usize, usize)>
where
    L: LoadKernel<I, 1>,
    T: WellFormedFloatElement,
    S: FloatSimd<T>,
{
    let mut idx = 0usize;
    let len = data.iter().map(|d| d.len()).min().unwrap_or(0);

    if len == 0 {
        return None;
    }

    let ptrs = data.map(|d| d.as_ptr());

    let mut min_vals = Vector::MAX;
    let mut max_vals = Vector::MIN;

    let mut min_indices = Vector::<S::uxN>::ZERO;
    let mut max_indices = Vector::<S::uxN>::ZERO;

    let mut curr_indices = Vector::<S::uxN>::indexed();

    let incr = Vector::<S::uxN>::offset();

    let lane_width = <S::NativeWidth as thermite::generic_array::typenum::Unsigned>::USIZE;

    while (len - idx) >= lane_width {
        let value = unsafe { loader.load::<Vector<S::fxN>>(ptrs.map(|p| p.add(idx)))[0] };

        let min_mask = value.cmp_lt(min_vals);
        let max_mask = value.cmp_gt(max_vals);

        min_vals = min_mask.select(min_vals, value);
        min_indices = min_mask.select(min_indices, curr_indices);

        max_vals = max_mask.select(max_vals, value);
        max_indices = max_mask.select(max_indices, curr_indices);

        curr_indices += incr;
        idx += lane_width;
    }

    let mut min_value = min_vals.extract::<0>();
    let mut min_index = min_indices.extract::<0>();

    let mut max_value = max_vals.extract::<0>();
    let mut max_index = max_indices.extract::<0>();

    for i in 1..lane_width {
        let lane_min_value = min_vals[i];
        let lane_min_index = min_indices[i];

        let lane_max_value = max_vals[i];
        let lane_max_index = max_indices[i];

        if lane_min_value < min_value {
            min_value = lane_min_value;
            min_index = lane_min_index;
        }

        if lane_max_value > max_value {
            max_value = lane_max_value;
            max_index = lane_max_index;
        }
    }

    let (Ok(mut min_index), Ok(mut max_index)) = (min_index.try_into(), max_index.try_into()) else {
        return None;
    };

    while idx < len {
        let value = unsafe { loader.load::<Vector<T>>(ptrs.map(|p| p.add(idx)))[0].0 };

        if value < min_value {
            min_value = value;
            min_index = idx;
        }

        if value > max_value {
            max_value = value;
            max_index = idx;
        }

        idx += 1;
    }

    Some((min_index, max_index))
}

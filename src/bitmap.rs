use alloc::vec::Vec;

#[derive(Default)]
pub struct Bitmap64 {
    map: Vec<u64>,
}

impl Bitmap64 {
    pub const BITS: usize = u64::BITS as usize;

    #[inline(always)]
    pub const fn word_filled_by(bit: bool) -> u64 {
        if bit { !0u64 } else { 0u64 }
    }

    const fn required_word_len(len: usize) -> usize {
        (len + Bitmap64::BITS as usize - 1) / Bitmap64::BITS
    }

    #[inline(always)]
    pub const fn word_index(i: usize) -> usize {
        i / Bitmap64::BITS
    }

    #[inline(always)]
    const fn word_offset(i: usize) -> u32 {
        (i % Bitmap64::BITS) as u32
    }

    #[inline(always)]
    const fn index_pair(i: usize) -> (usize, u32) {
        (Bitmap64::word_index(i), Bitmap64::word_offset(i))
    }

    pub const fn new_empty() -> Self {
        Self {
            map: vec![],
        }
    }

    pub fn new(len: usize) -> Self {
        Self {
            map: vec![Bitmap64::word_filled_by(false); Bitmap64::required_word_len(len)],
        }
    }

    pub fn len(&self) -> usize {
        self.map.len() * Bitmap64::BITS as usize
    }

    pub fn resize(&mut self, new_len: usize) {
        self.map.resize(Bitmap64::required_word_len(new_len), Bitmap64::word_filled_by(false));
    }

    /// Get 64bit bit-sequence 8-byte-aligned at 64bit-block index `wi`
    #[inline(always)]
    pub fn get_word(&self, wi: usize) -> u64 {
        if wi as usize >= self.map.len() {
            Bitmap64::word_filled_by(false)
        } else {
            self.map[wi]
        }
    }

    #[inline(always)]
    pub fn set_word(&mut self, wi: usize, bit: bool) {
        self.map[wi] = Bitmap64::word_filled_by(bit);
    }

    #[inline(always)]
    pub fn set_word_direct(&mut self, wi: usize, word: u64) {
        self.map[wi] = word;
    }

    /// Get 64bit bit-sequence beginning from index `i`
    #[cfg(test)] // Used for plus-operated double-array implementation
    #[inline(always)]
    pub fn get_word_from(&self, i: usize) -> u64 {
        let (q, r) = Bitmap64::index_pair(i);
        let a = self.get_word(q);
        if r == 0 {
            a
        } else {
            let b = self.get_word(q+1);
            a >> r | b << (64-r)
        }
    }

    #[inline(always)]
    pub fn get_bit(&self, i: usize) -> bool {
        let (q, r) = Bitmap64::index_pair(i);
        let w = self.get_word(q);
        let mask = 1u64 << r;
        w & mask != 0
    }

    #[inline(always)]
    pub fn set_bit(&mut self, i: usize, bit: bool) {
        let (q, r) = Bitmap64::index_pair(i);
        let mask = 1u64 << r;
        if bit {
            self.map[q as usize] |= mask;
        } else {
            self.map[q as usize] &= !mask;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bitmap() {
        let set = [1,3,7,13,47,101,255,1996,2022];
        let mut bm = Bitmap64::new(3000);
        // Random initialize
        for i in 0..3000 {
            if i % 7 % 2 == 0 {
                bm.set_bit(i, false);
            } else {
                bm.set_bit(i, true);
            }
        }
        let mut k = 0;
        for i in 0..3000 {
            if k == set.len() || i < set[k] {
                bm.set_bit(i, false);
            } else {
                bm.set_bit(i, true);
                k += 1;
            }
        }
        k = 0;
        for i in 0..3000 {
            if k == set.len() || i < set[k] {
                assert!(!bm.get_bit(i));
            } else {
                assert!(bm.get_bit(i));
                k += 1;
            }
        }
    }
}
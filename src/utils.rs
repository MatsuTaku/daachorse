use std::io::{self, Read, Write};

#[inline(always)]
pub fn read_u32<R>(mut rdr: R) -> io::Result<u32> where R: Read {
    let mut buf = [0; 4];
    rdr.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[inline(always)]
pub fn read_i32<R>(mut rdr: R) -> io::Result<i32> where R: Read {
    let mut buf = [0; 4];
    rdr.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

#[inline(always)]
pub fn read_u8<R>(mut rdr: R) -> io::Result<u8> where R: Read {
    let mut buf = [0];
    rdr.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline(always)]
pub fn write_u32<W>(mut wtr: W, data: u32) -> io::Result<()> where W: Write {
    wtr.write_all(&data.to_le_bytes())?;
    Ok(())
}

#[inline(always)]
pub fn write_i32<W>(mut wtr: W, data: i32) -> io::Result<()> where W: Write {
    wtr.write_all(&data.to_le_bytes())?;
    Ok(())
}

#[inline(always)]
pub fn write_u8<W>(mut wtr: W, data: u8) -> io::Result<()> where W: Write {
    wtr.write_all(&[data])?;
    Ok(())
}

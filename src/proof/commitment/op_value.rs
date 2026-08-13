//! Order-preserving VALUE encoding for numeric and date/time literals.
//!
//! A range query ("all objects with value `< 10`") must be checkable against COMMITTED data.
//! The catalog's dictionary ids are NOT ordered by value, so the scheme commits a separate
//! per-entry VALUE column: each supported fixed-width literal is encoded as an unsigned
//! integer whose numeric order MATCHES the literal's real order (negatives below positives,
//! dates as nanoseconds, etc.), and everything else (nodes, strings, unsupported datatypes)
//! gets a `0` sentinel. Because the encoding is order-preserving, a proof can compare these
//! committed integers directly. This module is that encoder: the numeric width table, the
//! signed-to-unsigned offset, and the (fiddly) decoders that read the Store's lexical
//! `xsd:date`/`xsd:dateTime` bytes into an i64-nanoseconds key. [`node_value_op_value`] is the
//! entry point; [`NODE_DATATYPE_TAG`] is the reserved discriminator marking a node (not a value).

use super::*;

/// Reserved datatype tag for a `Node` entry in the node/value datatype column.
/// `CanonicalDatatype` is `#[repr(u8)]` with 47 real variants (discriminants
/// `0..=46`, `AnySimpleType = 44`, `DateTimeInterval = 46`), so `255` is safely
/// distinct from every real discriminant. The later range gadget masks Node
/// rows out by a datatype-inequality check against this tag.
/// Reserved discriminator committed in the node/value datatype dictionary column.
/// Every node has this tag; every literal uses a `CanonicalDatatype` discriminant.
pub const NODE_DATATYPE_TAG: u64 = 255;

/// Padded length (a power of two `>= 2`, or `0` for an empty catalog) shared by
/// EVERY committed node/value dictionary column — `hash`, `id`, `value`,
/// `datatype`. Single source of truth so the four columns are the exact same
/// length and stay row-aligned (a range proof opens `value`/`datatype` at the
/// SAME index a membership proof opens `hash`/`id`).
pub(super) fn node_value_dict_padded_len(catalog_len: usize) -> io::Result<usize> {
    if catalog_len == 0 {
        return Ok(0);
    }
    padded_domain_for_rows(catalog_len).map(|(padded, _)| padded)
}

/// Fixed byte width of a SUPPORTED fixed-width order-preserving numeric
/// datatype's `to_lexical` encoding, or `None` for any other datatype
/// (variable-width like `String`/`Decimal`/`BigInt`, or non-range types).
///
/// CONFIRMED against tdb-succinct `src/tfc/datatypes.rs` (git rev ec15199,
/// the pinned dep): every listed type's `ToLexical::to_lexical` writes exactly
/// `width` bytes, and the raw bytes are ALREADY the order-preserving encoding,
/// so a big-endian unsigned read of the fixed-width bytes is monotone in value:
///   * `Boolean` -> 1 byte `vec![0]`/`vec![1]`         (false=0 < true=1).
///   * `UInt8/16/32/64` -> 1/2/4/8 bytes, `write_uN::<BigEndian>` (raw BE).
///   * `Int8/16/32/64`  -> 1/2/4/8 bytes, offset-binary: `to_lexical` writes
///     `(1<<(8*width-1)) ^ (self as uN)` big-endian, so `min` maps to all-zero
///     and BE-reading the bytes recovers a monotone unsigned code.
///   * `Float32/64` -> 4/8 bytes, IEEE total-order: `to_lexical` writes
///     `bits ^ 0x8000..` for non-negative and `bits ^ 0xffff..` for negative,
///     big-endian — the standard monotone transform over finite floats.
pub(super) fn order_preserving_numeric_width(datatype: CanonicalDatatype) -> Option<usize> {
    use CanonicalDatatype::*;
    Some(match datatype {
        Boolean => 1,
        UInt8 | Int8 => 1,
        UInt16 | Int16 => 2,
        UInt32 | Int32 | Float32 => 4,
        UInt64 | Int64 | Float64 => 8,
        // `Date`/`DateTime` are order-preserving at width 8, but — unlike every
        // entry above — their native `lexical` is NOT the OP encoding (tdb-succinct
        // stores dateTime as variable-width sign+bigint-seconds+variable-nanos, and
        // date as a `year:i64,month:u8,day:u8,offset:i16` struct). Their OP value is
        // DERIVED (decode → chronological key → 8-byte offset-binary) in
        // `node_value_op_value`, which special-cases these two BEFORE the generic
        // fixed-width read below. This function reports the OP *width* (8, the range
        // gadget's `Int64` convention); it does NOT claim the lexical is 8 bytes.
        // `DateTimeStamp`/`DateTimeInterval`/`Time`/`Duration`/g* temporal types are
        // deliberately NOT here (they reject-close to the `0` sentinel): the WOQL AST
        // cannot even name a `DateTimeStamp` constant, and adding them would perturb
        // the committed OP column of any layer holding such values.
        Date | DateTime => 8,
        _ => return None,
    })
}

/// The shared 8-byte OFFSET-BINARY encoding of a signed-integer key `n` as an
/// `ArkFr` — the SAME code the store's `Int64` path emits: `to_lexical` writes
/// `(1<<63) ^ (n as u64)` big-endian, and a big-endian unsigned read recovers that
/// value, mapping `i64::MIN → 0` and `i64::MAX → 2^64-1` monotonically. Routing the
/// date keys through this convention lets the AUDITED range gadget (sign-bit
/// bit-decomposition, `b_bits = 64`) compare them UNCHANGED. The datatype-tag column
/// still carries the `Date`/`DateTime` tag, so the gadget masks date rows to a date
/// query only — there is no cross-type comparison with the numeric `Int64` values
/// that happen to share this code.
pub(super) fn op_offset_binary_i64(n: i64) -> ArkFr {
    <ArkFr as DoryField>::from_u64((n as u64) ^ (1u64 << 63))
}

/// Map a decoded UTC `NaiveDateTime` to its i64 nanoseconds-since-epoch key,
/// SATURATING outside the representable i64-nanoseconds window (≈ 1677-09-21 ..
/// 2262-04-11). Saturating by chronological side keeps EVERY out-of-window date
/// correctly ordered against any in-window comparison constant `c`: a far-future date
/// (`i64::MAX`) is `> c` and a far-past date (`i64::MIN`) is `< c`. Since a range
/// filter only ever compares a data value to `c` (never two data values to each
/// other), the fact that two far dates tie is harmless.
pub(super) fn datetime_to_saturating_nanos(ndt: chrono::NaiveDateTime) -> i64 {
    let utc = ndt.and_utc();
    // `timestamp()` (seconds) always fits i64 for any `NaiveDateTime`, so its sign is
    // a safe far-past/far-future discriminator when the nanos overflow i64.
    utc.timestamp_nanos_opt()
        .unwrap_or(if utc.timestamp() >= 0 { i64::MAX } else { i64::MIN })
}

// --- Defensive native date/dateTime OP decoding ------------------------------------
//
// `VerifierCommitmentRef::validate()` recomputes `node_value_op_value` over EVERY
// committed catalog row of a self-contained, ADVERSARY-SUPPLIED proof. The native
// tdb-succinct decoders (`storage_to_datetime`, `Date::from_lexical`, and the
// integer/decimal helpers they call) `.unwrap()`/`.expect()`/index a `Buf`, so a
// crafted lexical PANICS them — crashing the honest verifier (a trustless-system
// DoS). We therefore decode the storage bytes ourselves so that ANY byte sequence
// yields either the correct `i64` OP key (`Ok`) or a clean `io::Error` — never a
// panic. For every well-formed, store-produced lexical the key is byte-identical to
// the native decoder; only well-formed-but-out-of-window instants saturate (the
// audited design) and only MALFORMED/undecodable lexicals become `Err`.
//
// Size-prefix bit layout of tdb-succinct's `bigint` storage (see integer.rs).
pub(super) const DT_FIRST_SIGN: u8 = 0b1000_0000;
pub(super) const DT_FIRST_CONTINUATION: u8 = 0b0100_0000;
pub(super) const DT_FIRST_MASK: u8 = 0b0011_1111;
pub(super) const DT_CONTINUATION: u8 = 0b1000_0000;
pub(super) const DT_BASE_MASK: u8 = 0b0111_1111;

/// Consume one byte from `cursor`, or fail closed on a truncated lexical.
pub(super) fn dt_take_u8(cursor: &mut &[u8]) -> io::Result<u8> {
    match cursor.split_first() {
        Some((byte, rest)) => {
            *cursor = rest;
            Ok(*byte)
        }
        None => Err(invalid("truncated date/dateTime lexical")),
    }
}

/// Defensive re-implementation of tdb-succinct's `size_decode`: read the
/// variable-length size prefix, returning `(sign_is_non_negative, magnitude_byte_count)`.
/// Uses SATURATING arithmetic so an adversarial run of continuation bytes cannot
/// overflow-panic (`size <<= 7` panics on `u32` overflow in a debug build); the
/// (possibly saturated) count is only ever an upper bound, since the magnitude read
/// below fails closed the moment the buffer runs out.
pub(super) fn dt_decode_size_prefix(cursor: &mut &[u8]) -> io::Result<(bool, u64)> {
    let mut size: u64 = 0;
    let mut sign = true;
    let mut first = true;
    loop {
        let raw = dt_take_u8(cursor)?;
        if first {
            sign = raw & DT_FIRST_SIGN != 0;
            let v = if sign { raw } else { !raw };
            let val = u64::from(v & DT_FIRST_MASK);
            if v & DT_FIRST_CONTINUATION == 0 {
                return Ok((sign, val));
            }
            size = size.saturating_add(val);
            first = false;
        } else {
            let v = if sign { raw } else { !raw };
            let val = u64::from(v & DT_BASE_MASK);
            if v & DT_CONTINUATION == 0 {
                return Ok((sign, size.saturating_add(val)));
            }
            size = size.saturating_add(val);
        }
        size = size.saturating_mul(128); // `<< 7`, saturating to dodge the debug overflow panic
    }
}

/// Defensive re-implementation of `storage_to_bigint_and_sign` specialised to the
/// dateTime seconds field: decode the signed magnitude, returning the sign and the
/// `i64` value, or `None` when the magnitude falls OUTSIDE the `i64` range (an
/// out-of-window instant the caller saturates). Fails closed on a truncated magnitude.
/// The tdb-succinct original `.expect("…surprisingly large number of seconds")`s here.
pub(super) fn dt_decode_seconds(cursor: &mut &[u8]) -> io::Result<(bool, Option<i64>)> {
    let (is_pos, size) = dt_decode_size_prefix(cursor)?;
    if size == 0 {
        return Ok((is_pos, Some(0)));
    }
    // A positive value leaves `i64` once the magnitude exceeds `i64::MAX`; a negative
    // one once it exceeds `i64::MAX + 1` (`-2^63 == i64::MIN` is representable).
    let limit: u128 = if is_pos {
        i64::MAX as u128
    } else {
        (i64::MAX as u128) + 1
    };
    let mut acc: u128 = 0;
    let mut overflow = false;
    for _ in 0..size {
        let byte = dt_take_u8(cursor)?;
        let byte = if is_pos { byte } else { !byte };
        if !overflow {
            acc = (acc << 8) | u128::from(byte);
            if acc > limit {
                // Stop accumulating (keeping `acc` bounded) but keep consuming the
                // remaining magnitude bytes so a truncation is still caught below.
                overflow = true;
            }
        }
    }
    if overflow {
        return Ok((is_pos, None));
    }
    let magnitude = acc as i128;
    let value = if is_pos { magnitude } else { -magnitude };
    match i64::try_from(value) {
        Ok(seconds) => Ok((is_pos, Some(seconds))),
        // Unreachable given `limit`, but fail safe (never panic) rather than trust it.
        Err(_) => Ok((is_pos, None)),
    }
}

/// Decode one centary-BCD value (tdb-succinct `centary_decimal_decode`), failing
/// closed on the `0` byte whose native `i - 1` would underflow-panic.
pub(super) fn dt_centary_decode(value: u8) -> io::Result<String> {
    if value == 0 {
        return Err(invalid("malformed dateTime fraction byte"));
    }
    let j = value - 1;
    if j % 11 == 0 {
        Ok(format!("{}", j / 11))
    } else {
        // `j >= 1` in this branch, so `j - d - 1 >= 0` (no u8 underflow).
        let d = j / 11;
        let num = j - d - 1;
        Ok(format!("{num:02}"))
    }
}

/// Defensive re-implementation of `decode_fraction`: read the sub-second fraction
/// digits (bit-inverted when the instant is negative). Consumes the WHOLE fraction
/// field — its terminal byte clears the continuation bit — so the caller can require
/// the lexical to be fully consumed. Fails closed on truncation, a malformed digit, or
/// a fraction longer than the 9 digits a nanosecond count can ever need (which would
/// underflow the native `9 - fraction.len()`).
pub(super) fn dt_decode_fraction(cursor: &mut &[u8], is_pos: bool) -> io::Result<String> {
    // The fraction field is ALWAYS at least one byte (a `0x00` "false zero" when there
    // is no fraction); its absence is a truncated lexical.
    let mut raw = dt_take_u8(cursor)?;
    let first = if is_pos { raw } else { !raw };
    if first == 0x00 {
        return Ok(String::new());
    }
    let mut digits = String::new();
    loop {
        let byte = if is_pos { raw } else { !raw };
        let chunk = dt_centary_decode(byte >> 1)?;
        let single = chunk.len() == 1;
        digits.push_str(&chunk);
        if single || byte & 1 == 0 {
            break;
        }
        if digits.len() > 9 {
            return Err(invalid("dateTime fraction exceeds 9 digits"));
        }
        raw = dt_take_u8(cursor)?;
    }
    Ok(digits)
}

/// Convert decoded fraction digits (`<= 9`) to a nanosecond count, mirroring
/// `storage_to_datetime`'s right-zero-pad-to-9 then parse. Fails closed on anything
/// longer than 9 digits or non-numeric (never for a store-produced fraction).
pub(super) fn dt_fraction_to_nanos(digits: &str) -> io::Result<u32> {
    if digits.len() > 9 {
        return Err(invalid("dateTime fraction exceeds 9 digits"));
    }
    let mut padded = String::with_capacity(9);
    padded.push_str(digits);
    for _ in 0..(9 - digits.len()) {
        padded.push('0');
    }
    padded
        .parse::<u32>()
        .map_err(|_| invalid("dateTime fraction is not a valid nanosecond count"))
}

/// Decode a native tdb-succinct `xsd:dateTime` `lexical` to its OP i64 key. WOQL folds
/// the timezone offset into the instant at parse time (`remove_date_time_offset`) and
/// stores a bare UTC `NaiveDateTime`, and compares two such values chronologically to
/// the nanosecond (`date_time/7` standard order) — exactly the epoch-nanoseconds order
/// this key reproduces. The storage bytes are decoded DEFENSIVELY (see the block
/// above): a bigint-seconds value outside `i64`, or an instant outside chrono's
/// representable range, SATURATES to `i64::MIN/MAX` by sign (the audited saturating
/// design); a truncated/over-long/otherwise-malformed lexical FAILS CLOSED with an
/// `io::Error` rather than panicking the verifier. For every store-produced lexical the
/// key is byte-identical to `storage_to_datetime` + `datetime_to_saturating_nanos`.
pub(super) fn datetime_lexical_to_op_i64(lexical: &[u8]) -> io::Result<i64> {
    use chrono::{DateTime, Utc};
    let mut cursor: &[u8] = lexical;
    let (is_pos, seconds) = dt_decode_seconds(&mut cursor)?;
    let fraction = dt_decode_fraction(&mut cursor, is_pos)?;
    // A well-formed dateTime lexical is exactly the seconds prefix + fraction field;
    // any trailing bytes are malformed.
    if !cursor.is_empty() {
        return Err(invalid("dateTime lexical has trailing bytes"));
    }
    let seconds = match seconds {
        Some(seconds) => seconds,
        // Bigint seconds beyond i64 ⇒ far outside the i64-nanoseconds window; saturate.
        None => return Ok(if is_pos { i64::MAX } else { i64::MIN }),
    };
    let nanos = if fraction.is_empty() {
        0
    } else {
        dt_fraction_to_nanos(&fraction)?
    };
    // Reconstruct the SAME instant `storage_to_datetime` would (checked constructors),
    // then reuse `datetime_to_saturating_nanos` for a byte-identical key. A negative
    // instant with a fraction is stored as `(seconds - 1, 1e9 - nanos)`.
    let instant = if fraction.is_empty() || is_pos {
        DateTime::<Utc>::from_timestamp(seconds, nanos)
    } else {
        seconds
            .checked_sub(1)
            .and_then(|s| DateTime::<Utc>::from_timestamp(s, 1_000_000_000 - nanos))
    };
    match instant {
        Some(instant) => Ok(datetime_to_saturating_nanos(instant.naive_utc())),
        // Fits i64 seconds but beyond chrono's representable range ⇒ far past/future,
        // outside the i64-nanoseconds window; saturate by sign.
        None => Ok(if is_pos { i64::MAX } else { i64::MIN }),
    }
}

/// Decode a native tdb-succinct `xsd:date` `lexical` to its OP i64 key. WOQL compares a
/// stored `xsd:date` as the Prolog compound `date(Year,Month,Day,Offset)` in standard
/// order — i.e. `(year, month, day, offset)`-LEXICAL, with the timezone offset as a
/// LEAST-significant tiebreaker that is never applied to shift the day (a larger offset
/// sorts later; `"2020-01-01"` and `"2020-01-01Z"` both have offset 0 and compare
/// equal). We reproduce THAT order — not a pure instant — with a key strictly monotone
/// in `(year, month, day, offset)`:
///
/// ```text
///   key = midnight_utc_nanos(year, month, day)  +  offset
/// ```
///
/// Consecutive calendar days differ by `86_400 * 1e9 ≈ 8.64e13` nanoseconds, which
/// dwarfs `|offset|` (a tdb-succinct `Date.offset` is an `i16`, ≤ 32767 in magnitude),
/// so the offset can only ever break ties among equal `(Y,M,D)` — exactly like
/// `compare/3`. A comparison constant always has offset 0 (the compiler builds it so),
/// so the tiebreaker is only exercised by DATA rows, keeping the constant's own key a
/// pure midnight instant.
pub(super) fn date_lexical_to_op_i64(lexical: &[u8]) -> io::Result<i64> {
    use chrono::NaiveDate;
    // A store `xsd:date` lexical is EXACTLY the 12-byte `Date` encoding —
    //   year:i64 (offset-binary BE, 8) | month:u8 (1) | day:u8 (1) | offset:i16 (offset-binary BE, 2).
    // Decode the fixed fields directly instead of `Date::from_lexical`, which reads a
    // `Buf` and PANICS on a short buffer; a wrong length is a truncated or over-long
    // (i.e. malformed) lexical ⇒ fail closed. These fixed-width reads reproduce
    // `Date::from_lexical` byte-for-byte for every valid 12-byte lexical (the same
    // offset-binary transform the store's `i64`/`i16` `to_lexical` applies).
    let [y0, y1, y2, y3, y4, y5, y6, y7, month, day, o0, o1]: [u8; 12] = lexical
        .try_into()
        .map_err(|_| invalid("xsd:date lexical is not exactly 12 bytes"))?;
    let year = (u64::from_be_bytes([y0, y1, y2, y3, y4, y5, y6, y7]) ^ (1u64 << 63)) as i64;
    let offset = (u16::from_be_bytes([o0, o1]) ^ (1u16 << 15)) as i16;
    let base = match i32::try_from(year) {
        Ok(year) => match NaiveDate::from_ymd_opt(year, u32::from(month), u32::from(day)) {
            // Midnight is always a valid time-of-day; still constructed via the checked
            // `and_hms_opt` (never `.expect`) so no input can panic the verifier.
            Some(date) => datetime_to_saturating_nanos(
                date.and_hms_opt(0, 0, 0)
                    .ok_or_else(|| invalid("xsd:date midnight is not constructible"))?,
            ),
            // A well-formed store `Date` always has a valid calendar day; an
            // out-of-range month/day here means the committed lexical is corrupt —
            // fail closed rather than guess a key.
            None => return Err(invalid("xsd:date lexical has an out-of-range month or day")),
        },
        // A year beyond `i32` is astronomically far past/future and unambiguously
        // outside the i64-nanoseconds window; saturate by its sign (year ≥ 0 ⇒ far
        // future). Here the sign of the YEAR is a sound discriminator precisely because
        // such a year is enormous.
        Err(_) => return Ok(if year >= 0 { i64::MAX } else { i64::MIN }),
    };
    // Apply the offset tiebreaker ONLY to an in-window base. A saturated boundary key
    // (a far date, or the exact representable min/max) stays pinned so the offset can
    // never nudge a far date back across an in-window constant. Real in-window dates
    // sit at midnight, ≥ ~8.5e13 ns from either extreme, so `saturating_add` here never
    // actually saturates for them — it is a guard for the boundary case only.
    Ok(match base {
        i64::MIN | i64::MAX => base,
        _ => base.saturating_add(i64::from(offset)),
    })
}

/// The order-preserving unsigned-integer encoding of a catalog entry's object
/// value as an `ArkFr`, or the `0` sentinel for entries that are never
/// range-queried (`Node`s, and `String`/variable-width or otherwise
/// unsupported datatypes). For a supported fixed-width numeric type the store's
/// `to_lexical` bytes ARE the OP encoding (see `order_preserving_numeric_width`),
/// so we big-endian read the fixed-width bytes into `[0, 2^{8*width})`. Widths
/// are `<= 8`, so the value always fits a `u64`. A supported type whose lexical
/// width does NOT match the expected fixed width is treated as corruption and
/// fails closed (this cannot happen for store-produced values).
///
/// `Date`/`DateTime` are the sole EXCEPTION: they are order-preserving but their
/// native lexical is NOT the OP encoding, so they are special-cased below —
/// decoded to a chronological i64 key and re-encoded via [`op_offset_binary_i64`].
/// This is `pub` so the woql2 compiler can compute a range constant's OP value
/// through this EXACT function (single source of truth: the compiler's `c` is then
/// byte-identical to the committed dict value column by construction).
pub fn node_value_op_value(object: &CanonicalObject) -> io::Result<ArkFr> {
    let (datatype, lexical) = match object {
        CanonicalObject::Value { datatype, lexical } => (*datatype, lexical),
        CanonicalObject::Node(_) => return Ok(<ArkFr as DoryField>::zero()),
    };
    // Date/DateTime: decode the native storage bytes to a chronological i64 key and
    // re-encode as the shared offset-binary code. Handled BEFORE the generic
    // fixed-width read because their `lexical` is variable-width (so the width check
    // below would reject it), even though `order_preserving_numeric_width` reports 8.
    match datatype {
        CanonicalDatatype::DateTime => {
            return Ok(op_offset_binary_i64(datetime_lexical_to_op_i64(lexical)?));
        }
        CanonicalDatatype::Date => {
            return Ok(op_offset_binary_i64(date_lexical_to_op_i64(lexical)?));
        }
        _ => {}
    }
    let width = match order_preserving_numeric_width(datatype) {
        Some(width) => width,
        None => return Ok(<ArkFr as DoryField>::zero()),
    };
    if lexical.len() != width {
        return Err(invalid(
            "order-preserving numeric literal has unexpected lexical width",
        ));
    }
    let mut buf = [0u8; 8];
    buf[8 - width..].copy_from_slice(lexical);
    Ok(<ArkFr as DoryField>::from_u64(u64::from_be_bytes(buf)))
}


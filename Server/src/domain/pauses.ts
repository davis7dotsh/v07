/**
 * Chooses where to cut a recording that is still being uploaded, so finished
 * speech can be transcribed before the take ends.
 *
 * `probabilities` are Silero speech probabilities, one per VAD window, starting
 * at the first untranscribed frame. Speech uses Silero's own hysteresis: it
 * starts at 0.5 and ends below 0.35. A candidate pause is silence with speech on
 * both sides. Rather than a fixed pause length, a pause qualifies when it is at
 * least as long as the median pause in the same stretch of audio, so cuts land
 * on the speaker's longer breaks (usually sentence or clause boundaries).
 *
 * Returns the window index at the middle of the first qualifying pause that
 * leaves at least `minimumWindows` before the cut, or undefined to wait for
 * more audio. Once `maximumWindows` have accumulated, the longest candidate
 * pause is accepted instead.
 */
export function findPause(
  probabilities: readonly number[],
  minimumWindows: number,
  maximumWindows: number,
) {
  const pauses: { start: number; end: number }[] = [];
  let speaking = false;
  let silenceStart: number | undefined;
  for (const [index, probability] of probabilities.entries()) {
    if (probability >= 0.5) {
      if (!speaking && silenceStart !== undefined) pauses.push({ start: silenceStart, end: index });
      speaking = true;
      silenceStart = undefined;
    } else if (probability < 0.35 && speaking) {
      speaking = false;
      silenceStart = index;
    }
  }
  if (!pauses.length) return;
  const lengths = pauses.map((pause) => pause.end - pause.start).sort((a, b) => a - b);
  const typical = lengths[Math.floor(lengths.length / 2)]!;
  const middle = (pause: { start: number; end: number }) => (pause.start + pause.end) >> 1;
  const candidates = pauses.filter((pause) => middle(pause) >= minimumWindows);
  const natural = candidates.find((pause) => pause.end - pause.start >= typical);
  if (natural) return middle(natural);
  if (probabilities.length < maximumWindows || !candidates.length) return;
  return middle(
    candidates.reduce((longest, pause) =>
      pause.end - pause.start > longest.end - longest.start ? pause : longest,
    ),
  );
}

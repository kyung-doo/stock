import express from "express";
import { execSync } from "child_process";
import YahooFinance from "yahoo-finance2";

const yahooFinance = new YahooFinance();
const app = express();
app.use(express.json());
app.use(express.static("public"));

// 코스피 전체 종목 가져오기
function getAllKOSPI() {
  try {
    const output = execSync("python get_kospi_all.py", { encoding: "utf-8" });
    return output
      .toString()
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((line) => {
        const [symbol, name] = line.split("|");
        return { symbol, name };
      });
  } catch (err) {
    console.error("Python 실행 오류:", err);
    return [];
  }
}

// Yahoo Finance에서 일별 데이터 가져오기 (개월 수 반영)
async function fetchDaily(symbol, months = 6) {
  try {
    const now = new Date();
    const start = new Date();
    start.setMonth(now.getMonth() - months);

    const result = await yahooFinance.chart(symbol, {
      period1: start.toISOString().slice(0, 10),
      period2: now.toISOString().slice(0, 10),
      interval: "1d",
    });

    if (!result.quotes || result.quotes.length === 0) return [];

    return result.quotes.map((d) => ({
      date: new Date(d.date).toISOString().slice(0, 10),
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));
  } catch (err) {
    console.error(`chart() failed for ${symbol}:`, err.message);
    return [];
  }
}

// 상승률 계산
function pctChange(a, b) {
  return b === 0 ? 0 : a / b - 1;
}

// ATR 계산
function calcATR(sd, days = 14) {
  if (sd.length < days + 1) return 0;
  let trs = [];
  for (let i = 1; i < days + 1; i++) {
    const prev = sd[i - 1],
      cur = sd[i];
    const tr = Math.max(
      cur.high - cur.low,
      Math.abs(cur.high - prev.close),
      Math.abs(cur.low - prev.close)
    );
    trs.push(tr);
  }
  return trs.reduce((a, b) => a + b, 0) / days;
}

// 거래량 기준으로 상위 N개 종목 가져오기 (병렬 처리)
async function getTopByVolume(limit = 100, months = 1) {
  const kospiList = getAllKOSPI();

  // 모든 종목에 대해 fetchDaily 병렬 실행
  const results = await Promise.all(
    kospiList.map(async ({ symbol, name }) => {
      const sd = await fetchDaily(symbol, months);
      if (sd.length === 0) return null;

      // 최근 기간 평균 거래량
      const avgVol = sd.reduce((sum, d) => sum + d.volume, 0) / sd.length;
      return { symbol, name, avgVol };
    })
  );

  // null 제거 후 거래량 내림차순 정렬
  const withVolume = results
    .filter(Boolean)
    .sort((a, b) => b.avgVol - a.avgVol);

  // 상위 limit개만 반환
  return withVolume.slice(0, limit);
}

// 추천 API
app.get("/top-risers", async (req, res) => {
  const topN = parseInt(req.query.topN || "20", 10);
  const months = parseInt(req.query.months || "3", 10); // ✅ 개월 수 파라미터 추가
  const symbols = await getTopByVolume(100, req.query.months);

  // 코스피 지수 데이터
  const kospiSymbol = "^KS11";
  const kospiDaily = await fetchDaily(kospiSymbol, months);
  const kospiLevel = kospiDaily.at(-1)?.close || 0;
  const window = Math.min(200, kospiDaily.length);
  const kospiMA200 =
    kospiDaily.slice(-window).reduce((s, d) => s + d.close, 0) / window;

  const tasks = symbols.map(async ({ symbol, name }) => {
    const sd = await fetchDaily(symbol, months);
    if (sd.length < 15 * months) return null; // 최소 데이터 확보

    const rMonths = pctChange(sd.at(-1).close, sd[0].close);
    const atr = calcATR(sd);
    const atrPct = (atr / sd.at(-1).close) * 100;

    // coDownRate
    const recentDays = 20;
    const kRecent = kospiDaily.slice(-recentDays - 1);
    const kospiDownDates = [];
    for (let i = 1; i < kRecent.length; i++) {
      if (kRecent[i].close < kRecent[i - 1].close)
        kospiDownDates.push(kRecent[i].date);
    }
    let coDown = 0;
    for (let i = 1; i < sd.length; i++) {
      if (kospiDownDates.includes(sd[i].date) && sd[i].close < sd[i - 1].close)
        coDown++;
    }
    const coDownRate = kospiDownDates.length
      ? coDown / kospiDownDates.length
      : 0;

    // 매수/매도 가격
    const stockPrice = sd.at(-1).close;
    let buyPrice, sellPrice;
    if (kospiLevel >= kospiMA200) {
      buyPrice = stockPrice * 0.995;
      sellPrice = stockPrice * 1.1;
    } else {
      buyPrice = stockPrice * 0.97;
      sellPrice = stockPrice * 1.05;
    }

    return { symbol, name, rMonths, atrPct, coDownRate, buyPrice, sellPrice };
  });

  const results = (await Promise.all(tasks)).filter(Boolean);
  results.sort((a, b) => b.rMonths - a.rMonths);

  res.json({ symbols: results.slice(0, topN) });
});

// 일별 데이터 API (차트용)
app.get("/daily", async (req, res) => {
  const symbol = req.query.symbol;
  const months = parseInt(req.query.months || "6", 10); // ✅ 개월 수 파라미터 추가
  const data = await fetchDaily(symbol, months);
  res.json(data);
});

app.listen(3000, () =>
  console.log("✅ Server running on http://localhost:3000")
);

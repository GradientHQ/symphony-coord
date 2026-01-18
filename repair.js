// repair.js
const { jsonrepair } = require('jsonrepair');

const fs = require('fs');
const input = fs.readFileSync(0, 'utf-8'); // 读取 stdin

try {
  const output = jsonrepair(input);
  console.log(output);
} catch (err) {
  console.error('Invalid JSON:', err.message);
  process.exit(1);
}

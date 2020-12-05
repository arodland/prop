/*!
 * Serialize all form data into a SearchParams string
 * (c) 2020 Chris Ferdinandi, MIT License, https://gomakethings.com
 * @param  {Node}   form The form to serialize
 * @return {String}      The serialized form data
 */
var serialize = function (form) {
	var arr = [];
	Array.prototype.slice.call(form.elements).forEach(function (field) {
		if (!field.name || field.disabled || ['file', 'reset', 'submit', 'button'].indexOf(field.type) > -1) return;
		if (field.type === 'select-multiple') {
			Array.prototype.slice.call(field.options).forEach(function (option) {
				if (!option.selected) return;
				arr.push(encodeURIComponent(field.name) + '=' + encodeURIComponent(option.value));
			});
			return;
		}
		if (['checkbox', 'radio'].indexOf(field.type) >-1 && !field.checked) return;
		arr.push(encodeURIComponent(field.name) + '=' + encodeURIComponent(field.value));
	});
	return arr.join('&');
};
